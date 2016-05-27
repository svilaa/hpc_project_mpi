//
//  convolution.c
//
//
//  Created by Josep Lluis Lerida on 11/03/15.
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing large images (>500MB)
// The 2D image is represented by 1D vector for chanel R, G and B. The convolution is applied to each channel separately.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "mpi.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define TRUE 1
#define FALSE 0

#define PARTITION_DATA 9

#define PARTITION_START_ROW 0
#define PARTITION_END_ROW 1
#define PARTITION_RANGE 2
#define PARTITION_START_CHUNK_ROW 3
#define PARTITION_END_CHUNK_ROW 4
#define PARTITION_CHUNK_RANGE 5
#define PARTITION_HALO_ROWS 6
#define PARTITION_OFFSET_ROWS 7
#define PARTITION_WIDTH 8

// Estructura per emmagatzemar el contingut d'una imatge.
struct imagenppm{
    int height;
    int width;
    char *comment;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* DataImage;

// Estructura per emmagatzemar el contingut d'un kernel.
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

static int imagesize, partitions, subpartitions, individualPartitions, halo, width, height;
static long position=0;
static double start, tstart=0, tend=0, tread=0, treadp=0, tcopy=0, tconv=0, tstore=0, tstorep=0, treadk=0,
tsending = 0, treceiving = 0, tparallel = 0, tstartParallel = 0;
static FILE *fpsrc=NULL,*fpdst=NULL;
static DataImage source=NULL, output=NULL;
static int sourceLength, outputLength;
static int numReceivedData;
static int i=0;
static int rank, nprocs;
static MPI_Status status;
static kernelData kern=NULL;
static int *partitionData, *processData;
static int c;

static int combinedPartitionsChunkRows, combinedPartitionsHaloRows,
           combinedPartitionsOffsetRows, combinedPartitionsPixels;

static int haloRows, chunkRows, offsetRows;
static int rowSize, pendingRows, pendingPartitions, range, chunkRange, startRow, endRow, startChunkRow, endChunkRow, usefulChunkRows;
static int hasBufferedPartition;
static int pendingProcessesToNotifyEnd;
static int processRequest, pendingPartitionsToDeliver, totalPixels, currentNumProcess;
static int currentIndividualPartition, currentPartition;
static int firstIndividualPartitionToWrite;
static int obtainedIndividualPartition, firstPixel, individualPartitionsToWrite;
static int currentIndividualPartitionToWrite, currentIndividualPartitionStart;
static int currentIndividualPartitionUsefulPixels;
static int pendingCurrentIndividualPartitions;
static int blockPetition = -1;

static int *individualPartitionAssignation, *individualPartitionCompletion;

int canWork, usefulPixels, offset;
int *R_in, *G_in, *B_in, *R_out, *G_out, *B_out;
int lastTotalPixels;

//Functions Definition
DataImage initimage(char* name, FILE **fp, int partitions, int halo);
DataImage duplicateImageData(DataImage src, int partitions, int halo);

int readImage(DataImage Img, FILE **fp, int dim, int halosize, long int *position);
int duplicateImageChunk(DataImage src, DataImage dst, int dim);
int initfilestore(DataImage img, FILE **fp, char* name, long *position);
int savingChunk(DataImage img, FILE **fp, int dim, int offset);
int convolve2D(int* inbuf, int* outbuf, int sizeX, int sizeY, float* kernel, int ksizeX, int ksizeY);
void freeImagestructure(DataImage *src);

//Open Image file and image struct initialization
DataImage initimage(char* name, FILE **fp,int partitions, int halo){
    char c;
    char comment[300];
    int i=0,chunk=0;
    DataImage img=NULL;
    
    /*Se habre el fichero ppm*/

    if ((*fp=fopen(name,"r"))==NULL){
        perror("Error: ");
    }
    else{
        //Memory allocation
        img=(DataImage) malloc(sizeof(struct imagenppm));

        //Reading the first line: Magical Number "P3"
        fscanf(*fp,"%c%d ",&c,&(img->P));
        
        //Reading the image comment
        while((c=fgetc(*fp))!= '\n'){comment[i]=c;i++;}
        comment[i]='\0';
        //Allocating information for the image comment
        img->comment = malloc(strlen(comment)*sizeof(char));
        strcpy(img->comment,comment);
        //Reading image dimensions and color resolution
        fscanf(*fp,"%d %d %d",&img->width,&img->height,&img->maxcolor);
        chunk = img->width*img->height / partitions;
        //We need to read an extra row.
        chunk = chunk + img->width * halo;
        chunk += img->width;
        if ((img->R=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
        if ((img->G=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
        if ((img->B=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

//Duplicate the Image struct for the resulting image
DataImage duplicateImageData(DataImage src, int partitions, int halo){
    char c;
    char comment[300];
    unsigned int imageX, imageY;
    int i=0, chunk=0;
    //Struct memory allocation
    DataImage dst=(DataImage) malloc(sizeof(struct imagenppm));

    //Copying the magic number
    dst->P=src->P;
    //Copying the string comment
    dst->comment = malloc(strlen(src->comment)*sizeof(char));
    strcpy(dst->comment,src->comment);
    //Copying image dimensions and color resolution
    dst->width=src->width;
    dst->height=src->height;
    dst->maxcolor=src->maxcolor;
    chunk = dst->width*dst->height / partitions;
    //We need to read an extra row.
    chunk = chunk + src->width * halo;
    chunk += src->width;
    if ((dst->R=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
    if ((dst->G=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
    if ((dst->B=malloc(chunk*sizeof(int))) == NULL) {return NULL;}
    return dst;
}

//Read the corresponding chunk from the source Image
int readImage(DataImage img, FILE **fp, int dim, int halosize, long *position){
    int i=0, k=0,haloposition=0;
    if (fseek(*fp,*position,SEEK_SET))
        perror("Error: ");
    haloposition = dim-(img->width*halosize*2);
    for(i=0;i<dim;i++) {
        // When start reading the halo store the position in the image file
        if (halosize != 0 && i == haloposition) *position=ftell(*fp);
        fscanf(*fp,"%d %d %d ",&img->R[i],&img->G[i],&img->B[i]);
        k++;
    }
//    printf ("Readed = %d pixels, posicio=%lu\n",k,*position);
    return 0;
}

//Duplication of the  just readed source chunk to the destiny image struct chunk
int duplicateImageChunk(DataImage src, DataImage dst, int dim){
    int i=0;
    
    for(i=0;i<dim;i++){
        dst->R[i] = src->R[i];
        dst->G[i] = src->G[i];
        dst->B[i] = src->B[i];
    }
//    printf ("Duplicated = %d pixels\n",i);
    return 0;
}

// Open kernel file and reading kernel matrix. The kernel matrix 2D is stored in 1D format.
kernelData readKernel(char* name){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;
    
    /*Opening the kernel file*/
    fp=fopen(name,"r");
    if(!fp){
        perror("Error: ");
    }
    else{
        //Memory allocation
        kern=(kernelData) malloc(sizeof(struct structkernel));
        
        //Reading kernel matrix dimensions
        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
        
        // Reading kernel matrix values
        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Open the image file with the convolution results
int initfilestore(DataImage img, FILE **fp, char* name, long *position){
    /*Se crea el fichero con la imagen resultante*/
    if ( (*fp=fopen(name,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }
    /*Writing Image Header*/
    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comment,img->width,img->height,img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image partition to the resulting file. dim is the exact size to write. offset is the displacement for avoid halos.
int savingChunk(DataImage img, FILE **fp, int dim, int offset){
    int i,k=0;
    //Writing image partition
    for(i=offset;i<dim+offset;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
//        if ((i+1)%6==0) fprintf(*fp,"\n");
        k++;
    }
//    printf ("Writed = %d pixels, dim=%d, offset=%d\n",k,dim, offset);
    return 0;
}

// This function free the space allocated for the image structure.
void freeImagestructure(DataImage *src){
    
    free((*src)->comment);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    
    free(*src);
}

///////////////////////////////////////////////////////////////////////////////
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
//
//
// signed integer (32bit) version:
///////////////////////////////////////////////////////////////////////////////
int convolve2D(int* in, int* out, int dataSizeX, int dataSizeY,
               float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    int *inPtr, *inPtr2, *outPtr;
    float *kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;                             // to check boundary of input array
    int colMin, colMax;                             //
    float sum;                                      // temp accumulation buffer
    
    // check validity of params
    if(!in || !out || !kernel) return -1;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return -1;
    
    // find center position of kernel (half of kernel size)
    kCenterX = (int)kernelSizeX / 2;
    kCenterY = (int)kernelSizeY / 2;
    
    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;
    
    // start convolution
    for(i= 0; i < dataSizeY; ++i)                   // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;
        
        for(j = 0; j < dataSizeX; ++j)              // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;
            
            sum = 0;                                // set to 0 before accumulate
            
            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for(m = 0; m < kernelSizeY; ++m)        // kernel rows
            {
                // check if the index is out of bound of input array
                if(m <= rowMax && m > rowMin)
                {
                    for(n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if(n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;
                        
                        ++kPtr;                     // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;            // out of bound, move to next row of kernel
                
                inPtr -= dataSizeX;                 // move input data 1 raw up
            }
            
            // convert integer number
            if(sum >= 0) *outPtr = (int)(sum + 0.5f);
//            else *outPtr = (int)(sum - 0.5f)*(-1);
            // For using with image editors like GIMP or others...
//            else *outPtr = (int)(sum - 0.5f);
            // For using with a text editor that read ppm images like libreoffice or others...
            else *outPtr = 0;
            kPtr = kernel;                          // reset kernel to (0,0)
            inPtr = ++inPtr2;                       // next input
            ++outPtr;                               // next output
        }
    }
    
    return 0;
}

void RGBconvolve2D(int* R_in, int* R_out, int* G_in, int* G_out, int* B_in, int* B_out, int dataSizeX, int dataSizeY,
               float* kernel, int kernelSizeX, int kernelSizeY) {
    convolve2D( R_in,
                R_out,
                dataSizeX,
                dataSizeY,
                kernel,
                kernelSizeX,
                kernelSizeY);
    convolve2D( G_in,
                G_out,
                dataSizeX,
                dataSizeY,
                kernel,
                kernelSizeX,
                kernelSizeY);
    convolve2D( B_in,
                B_out,
                dataSizeX,
                dataSizeY,
                kernel,
                kernelSizeX,
                kernelSizeY);
}

void assignPartitionData() {
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_START_ROW] = startRow;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_END_ROW] = endRow;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_RANGE] = usefulChunkRows;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_START_CHUNK_ROW] = startChunkRow;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_END_CHUNK_ROW] = endChunkRow;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_CHUNK_RANGE] = chunkRange;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_HALO_ROWS] = haloRows;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_OFFSET_ROWS] = offsetRows;
    partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_WIDTH] = source->width;
}

int calculatePartitionData() {
    int maxRow = source->height - 1;

    for(currentIndividualPartition=0; currentIndividualPartition<subpartitions; currentIndividualPartition++) {
        range = pendingRows/pendingPartitions;
        pendingRows -= range;
        pendingPartitions--;
                
        if(c<individualPartitions-1){
            endRow = startRow + range - 1;
            usefulChunkRows = range;
        } else{
            endRow = source->height-1;
            usefulChunkRows = endRow - startRow + 1;
        }

        if (c==0) {
            haloRows  = halo/2;
            chunkRows = usefulChunkRows + haloRows;
            offsetRows = 0;

            startChunkRow = startRow;
            endChunkRow = endRow + haloRows;
            chunkRange = endChunkRow - startChunkRow + 1;
        }
        else if(c<individualPartitions-1) {
            haloRows  = halo;
            chunkRows = usefulChunkRows + haloRows;
            offsetRows = halo/2;

            startChunkRow = MAX(0, startRow - offsetRows);
            endChunkRow = endRow + halo/2;
            chunkRange = endChunkRow - startChunkRow + 1;
            haloRows = chunkRange - (endRow - startRow + 1);
            offsetRows = startRow - startChunkRow;
        }
        else {
            haloRows  = halo/2;
            chunkRows = usefulChunkRows + haloRows;
            offsetRows = halo/2;

            startChunkRow = MAX(0, startRow - offsetRows);
            endChunkRow = endRow;
            chunkRange = endChunkRow - startChunkRow + 1;
        }

        assignPartitionData();

        //printf("Partition %d individualPartition %d:\n\tstart: %d\n\tend: %d\n\trange: %d\n\tstartChunkRow: %d\n\tendChunkRow: %d\n\tchunkRange: %d\n\thaloRows: %d\n\toffsetRows: %d\n",
        //                currentPartition, currentIndividualPartition, startRow, endRow, range, startChunkRow, endChunkRow, chunkRange, haloRows, offsetRows);
        if(c<individualPartitions-1){
            startRow = endRow + 1;
        }

        c++;
    }
    return c;
}

void calculateCombinedPartitionData() {
    combinedPartitionsChunkRows = 0;
    for(currentIndividualPartition=0; currentIndividualPartition < subpartitions; currentIndividualPartition++) {
        combinedPartitionsChunkRows += partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_RANGE];
    }
    combinedPartitionsHaloRows = partitionData[0*PARTITION_DATA + PARTITION_OFFSET_ROWS] + // Upper halo rows
                                    partitionData[(subpartitions-1)*PARTITION_DATA + PARTITION_HALO_ROWS] - // Bottom halo rows
                                    partitionData[(subpartitions-1)*PARTITION_DATA + PARTITION_OFFSET_ROWS];
    combinedPartitionsOffsetRows = partitionData[0*PARTITION_DATA + PARTITION_OFFSET_ROWS];
            
    combinedPartitionsPixels = (combinedPartitionsChunkRows + combinedPartitionsHaloRows) * source->width;
}

void tryOuputRealloc() {
    if(outputLength < combinedPartitionsPixels) {
        output->R = realloc(output->R, combinedPartitionsPixels * sizeof(int));
        output->G = realloc(output->G, combinedPartitionsPixels * sizeof(int));
        output->B = realloc(output->B, combinedPartitionsPixels * sizeof(int));
        outputLength = combinedPartitionsPixels;
    }   
}

void trySourceRealloc() {
    if(sourceLength < totalPixels) {
        source->R = realloc(source->R, totalPixels * sizeof(int));
        source->G = realloc(source->G, totalPixels * sizeof(int));
        source->B = realloc(source->B, totalPixels * sizeof(int));
        sourceLength = totalPixels;
    }
}

int tryToObtainBufferedPartition() {
    if(!hasBufferedPartition && pendingCurrentIndividualPartitions > 0) {
        totalPixels = partitionData[currentIndividualPartition*PARTITION_DATA + PARTITION_CHUNK_RANGE] * source->width;
        
        start = MPI_Wtime();

        trySourceRealloc();

        //printf("Reading %d from image\n", totalPixels);
        if (readImage(source, &fpsrc, totalPixels, halo/2, &position)) {
            return -1;
        }
        treadp = treadp + (MPI_Wtime() - start);

        hasBufferedPartition = TRUE;
        pendingCurrentIndividualPartitions--;
        //printf("Readed image\n");
    }
    return 0;
}

int receiveFinishedPartition() {
    //printf("Partition finished\n");

    obtainedIndividualPartition = individualPartitionAssignation[currentNumProcess];
    individualPartitionCompletion[obtainedIndividualPartition] = TRUE;

    firstPixel = (partitionData[obtainedIndividualPartition*PARTITION_DATA + PARTITION_START_ROW] -
                  partitionData[0*PARTITION_DATA + PARTITION_START_ROW]) * source->width;

    //printf("Ready to receive data\n");
    //printf("Partition %d from %d\n", obtainedIndividualPartition, currentNumProcess);
    //printf("firstPixel %d\n", firstPixel);
    //printf("Receiving %d\n", processRequest);
    start = MPI_Wtime();
    MPI_Recv(&output->R[firstPixel], processRequest, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&output->G[firstPixel], processRequest, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&output->B[firstPixel], processRequest, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD, &status);
    treceiving = treceiving + (MPI_Wtime() - start);
    //printf("Received data\n");

    i = firstIndividualPartitionToWrite;
    while(i<subpartitions) {
        //printf("i: %d individualPartitionCompletion: %d\n", i, individualPartitionCompletion[i]);
        if(individualPartitionCompletion[i]) {
            i++;
        } else {
            break;
        }
    }

    //printf("firstIndividualPartitionToWrite %d\n", firstIndividualPartitionToWrite);

    individualPartitionsToWrite = i - firstIndividualPartitionToWrite;
    //printf("individualPartitionsToWrite %d\n", individualPartitionsToWrite);
    for(i = 0; i < individualPartitionsToWrite; i++) {
        currentIndividualPartitionToWrite = firstIndividualPartitionToWrite + i;
        currentIndividualPartitionStart = (partitionData[currentIndividualPartitionToWrite*PARTITION_DATA + PARTITION_START_ROW] - 
                                           partitionData[0*PARTITION_DATA + PARTITION_START_ROW]) * source->width;
        currentIndividualPartitionUsefulPixels = partitionData[currentIndividualPartitionToWrite*PARTITION_DATA + PARTITION_RANGE] * source->width;

        //printf("currentIndividualPartitionToWrite: %d\n", currentIndividualPartitionToWrite);
        //printf("currentIndividualPartitionStart: %d\n", currentIndividualPartitionStart);
        //printf("currentIndividualPartitionUsefulPixels: %d\n", currentIndividualPartitionUsefulPixels);
        start = MPI_Wtime();
        if (savingChunk(output, &fpdst, currentIndividualPartitionUsefulPixels, currentIndividualPartitionStart)) {
            perror("Error: ");
            //        free(source);
            //        free(output);
            return -1;
        }
        tstorep = tstorep + (MPI_Wtime() - start);
        //printf("Finished savingChunk\n");
    }

    firstIndividualPartitionToWrite += individualPartitionsToWrite;

    return 0;
}

void sendNewPartition() {
    //printf("A new partition can be sent\n");
    pendingPartitionsToDeliver--;
    hasBufferedPartition = FALSE;
    individualPartitionAssignation[currentNumProcess] = currentIndividualPartition;
    //printf("Asssigned partition %d to process %d\n", currentIndividualPartition, currentNumProcess);
    //printf("Sending partition\n");
                        
    start = MPI_Wtime();
    MPI_Send(&partitionData[currentIndividualPartition*PARTITION_DATA], PARTITION_DATA, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
    MPI_Send(source->R, totalPixels, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
    MPI_Send(source->G, totalPixels, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
    MPI_Send(source->B, totalPixels, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
    tsending = tsending + (MPI_Wtime() - start);
}

void notifyEnd() {
    //printf("No more partitions can be sent\n");
    pendingProcessesToNotifyEnd--;
    MPI_Send(NULL, 0, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
}

void blockSlave() {
    //printf("Blocking process\n");
    pendingProcessesToNotifyEnd--;
    MPI_Send(&blockPetition, 1, MPI_INT, currentNumProcess, 0, MPI_COMM_WORLD);
}

void newPartitionRequest() {
    //printf("pendingPartitionsToDeliver %d currentPartition %d\n", pendingPartitionsToDeliver, currentPartition);
    if(pendingPartitionsToDeliver > 0) { // A new partition can be sent
        sendNewPartition();
    } else if(pendingPartitionsToDeliver == 0 && currentPartition == partitions-1) { // No more partitions
        notifyEnd();
    } else { // No more subpartitions
        blockSlave();
    }
}

int manageRequests() {
    pendingProcessesToNotifyEnd = nprocs-1;
    currentIndividualPartition = 0;
    firstIndividualPartitionToWrite = 0;
    pendingCurrentIndividualPartitions = subpartitions;
    pendingPartitionsToDeliver = subpartitions;
    hasBufferedPartition = FALSE;

    while(pendingProcessesToNotifyEnd > 0) {
        tryToObtainBufferedPartition();

        //printf("Master ready to receive\n");

        MPI_Recv(&processRequest, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &numReceivedData);
        currentNumProcess = status.MPI_SOURCE;
        //printf("Received message from %d\n", currentNumProcess);

        if(numReceivedData != 0) { // Partition finished
            receiveFinishedPartition();
        } else { // A process requests a new partition
            newPartitionRequest();
        }

        if(!hasBufferedPartition) {
            currentIndividualPartition++;
        }
    }

    return 0;
}

void unlockAllSlaves() {
    //printf("Sending unlock messages\n");
    if(currentPartition < partitions-1) { // Unlock all the processes if more partitions are required
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //printf("Sent all the unlock messages\n");
}

int masterExecution() {
    c = 0;
    currentPartition = 0;
    startRow = 0;

    while(c<individualPartitions) {
        //printf("c %d; individualPartitions %d; subpartitions %d\n", c, individualPartitions, subpartitions);
        //printf("Starting a partition\n");

        for(i=0; i<subpartitions; i++) {
            individualPartitionCompletion[i] = FALSE;
        }

        c = calculatePartitionData();
        calculateCombinedPartitionData();
        tryOuputRealloc();
        manageRequests();

        if(partitions > 1) {
            unlockAllSlaves();
        }
        currentPartition++;
    }
    return 0;
}

void calculateIncomingPartitionSize() {
    //printf("---Process obtains new data\n");
    usefulPixels = processData[PARTITION_RANGE] * processData[PARTITION_WIDTH];
    totalPixels = processData[PARTITION_CHUNK_RANGE] * processData[PARTITION_WIDTH];
    offset = processData[PARTITION_OFFSET_ROWS] * processData[PARTITION_WIDTH];
    //printf("---Received remote offset %d\n", offset);
}

void trySlaveRealloc() {
    if(lastTotalPixels < totalPixels) {
        R_in = realloc(R_in, totalPixels * sizeof(int));
        G_in = realloc(G_in, totalPixels * sizeof(int));
        B_in = realloc(B_in, totalPixels * sizeof(int));
        R_out = realloc(R_out, totalPixels * sizeof(int));
        G_out = realloc(G_out, totalPixels * sizeof(int));
        B_out = realloc(B_out, totalPixels * sizeof(int));
        lastTotalPixels = totalPixels;
    }   
}

void receivePartition() {
    MPI_Recv(R_in, totalPixels, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(G_in, totalPixels, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(B_in, totalPixels, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
}

void SendConvolvedPartition() {
    //printf("---Sending finished data\n");
    //printf("---Sending %d usefulPixels\n", usefulPixels);
    //printf("---Remote offset %d\n", offset);
    MPI_Send(&usefulPixels, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&R_out[offset], usefulPixels, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&G_out[offset], usefulPixels, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&B_out[offset], usefulPixels, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void slaveExecution() {
    R_in = G_in = B_in = R_out = G_out = B_out = NULL;
    lastTotalPixels = 0;
    canWork = TRUE;
    while(canWork) {
        //printf("---Process sends petition\n");
        MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
        MPI_Recv(processData, PARTITION_DATA, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("---Answer\n");
        MPI_Get_count(&status, MPI_INT, &numReceivedData);
        if(numReceivedData == 0) { // Process can finish
            //printf("---Process must end\n");
            canWork = FALSE;
        } else {
            if(processData[0] == blockPetition) { // Block process until master unlock it
                //printf("---Process blocked\n");
                MPI_Barrier(MPI_COMM_WORLD);
                //printf("---Process unblocked\n");
            } else {
                calculateIncomingPartitionSize();
                trySlaveRealloc();
                receivePartition();

                RGBconvolve2D(R_in,R_out,G_in,G_out,B_in,B_out,
                    processData[PARTITION_WIDTH],
                    processData[PARTITION_CHUNK_RANGE],
                    kern->vkern,
                    kern->kernelX,
                    kern->kernelY);

                SendConvolvedPartition();
            }
        }
    }
    free(R_in); free(G_in); free(B_in); free(R_out); free(G_out); free(B_out);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    MPI_Init( &argc, &argv );
//    int headstored=0, imagestored=0, stored;
    
    if(argc != 6)
    {
        printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions> <subpartitions>\n", argv[0]);
        
        printf("\n\nError, Missing parameters:\n");
        printf("format: ./serialconvolution image_file kernel_file result_file\n");
        printf("- image_file : source image path (*.ppm)\n");
        printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
        printf("- result_file: result image path (*.ppm)\n");
        printf("- partitions : Image partitions\n");
        printf("- subpartitions per partition\n\n");
        MPI_Finalize();
        return -1;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // READING IMAGE HEADERS, KERNEL Matrix, DUPLICATE IMAGE DATA, OPEN RESULTING IMAGE FILE
    //////////////////////////////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////////////////////////////
    // MPI 
    //////////////////////////////////////////////////////////////////////////////////////////////////
     
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    if(nprocs < 2) {
        printf("Required at least 2 processes\n");
        return -1;
    }

    // Store number of partitions
    partitions = atoi(argv[4]);
    subpartitions = atoi(argv[5]);
    individualPartitions = partitions * subpartitions;
    if(rank==0) {
        partitionData = (int*)malloc(subpartitions*PARTITION_DATA*sizeof(int));
    } else {
        processData = (int*)malloc(PARTITION_DATA*sizeof(int));
    }

    ////////////////////////////////////////
    //Reading kernel matrix
    start = MPI_Wtime();

    tstart = start;
    if ( (kern = readKernel(argv[2]))==NULL) {
        //        free(source);
        //        free(output);
        MPI_Finalize();
        return -1;
    }
    //The matrix kernel define the halo size to use with the image. The halo is zero when the image is not partitioned.
    if (individualPartitions==0) {
        printf("partitions must be higher than 0.\n");
        MPI_Finalize();
        return -1;
    }
    if (individualPartitions==1) halo=0;
    else halo = (kern->kernelY/2)*2; // This operation subtract 1 if kernelY is odd

    treadk = treadk + (MPI_Wtime() - start);

    ////////////////////////////////////////
    //Reading Image Header. Image properties: Magical number, comment, size and color resolution.
    start = MPI_Wtime();

    if(rank == 0) {
        //Memory allocation based on number of partitions and halo size.
        if ( (source = initimage(argv[1], &fpsrc, individualPartitions, halo)) == NULL) {
            MPI_Finalize();
            return -1;
        }

        width = source->width;
        height = source->height;

        tread = tread + (MPI_Wtime() - start);
        
        //Duplicate the image struct.
        start = MPI_Wtime();
        if ( (output = duplicateImageData(source, partitions, halo)) == NULL) {
            MPI_Finalize();
            return -1;
        }

        sourceLength = (source->height/subpartitions) * source->width;
        outputLength = (source->height/partitions) * source->width;

        tcopy = tcopy + (MPI_Wtime() - start);
        
        ////////////////////////////////////////
        //Initialize Image Storing file. Open the file and store the image header.

        start = MPI_Wtime();
        if (initfilestore(output, &fpdst, argv[3], &position)!=0) {
            perror("Error: ");
            //        free(source);
            //        free(output);
            MPI_Finalize();
            return -1;
        }
        tstore = tstore + (MPI_Wtime() - start);
    }
        
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // CHUNK READING
    //////////////////////////////////////////////////////////////////////////////////////////////////
    
    if(rank==0) {
        individualPartitionAssignation = (int*)malloc(nprocs * sizeof(int));
        individualPartitionCompletion = (int*)malloc(subpartitions * sizeof(int));
    }

    if(rank == 0) {
        imagesize = source->height*source->width;
        rowSize  = source->height;
        //printf("%s ocupa %dx%d=%d pixels. Partitions=%d, halo=%d, partsize=%d pixels\n", argv[1], source->height, source->width, imagesize, partitions, halo, rowSize);
    }

    pendingRows = rowSize;
    pendingPartitions = individualPartitions;

    tstartParallel = MPI_Wtime();

    if(rank == 0) {
        masterExecution();
    } else {
        slaveExecution();
    }

    tparallel = (MPI_Wtime() - tstartParallel);
    tconv = tparallel - tstorep - treadp - tsending - treceiving;


    if(rank == 0) {
        fclose(fpsrc);
        fclose(fpdst);
        
        freeImagestructure(&source);
        freeImagestructure(&output);
    }

    
    if(rank == 0) {
        tend = MPI_Wtime();
        printf("Image  : %s\n", argv[1]);
        printf("Kernel : %s\n", argv[2]);
        printf("Output : %s\n", argv[3]);
        printf("ISizeX : %d\n", width);
        printf("ISizeY : %d\n", height);
        printf("kSizeX : %d\n", kern->kernelX);
        printf("kSizeY : %d\n", kern->kernelY);
        printf("%.6lf seconds elapsed for reading image file.\n", tread);
        printf("%.6lf seconds elapsed for reading image file in parallelism.\n", treadp);
        printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
        printf("%.6lf seconds elapsed for reading kernel matrix.\n", treadk);
        printf("%.6lf seconds elapsed for make the convolution.\n", tconv);
        printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
        printf("%.6lf seconds elapsed for writing the resulting image in parallelism.\n", tstorep);
        printf("%.6lf seconds elapsed\n", tend-tstart);
        printf("%.6lf seconds elapsed sending data.\n", tsending);
        printf("%.6lf seconds elapsed receiving data.\n", treceiving);
        printf("%.6lf seconds elapsed with parallelism.\n", tparallel);
        //<image>;<kernel>;<result>;<partitions>;<threads>
        //<reading>;<copying>;<reading-kernel>;<convolution>;<writing>;<total>
        //<sending-data>;<receiving-data>;<parallelism>
        printf("%s;%s;%s;%s;%d;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf;%.6lf\n",
            argv[1],argv[2],argv[3],argv[4],nprocs,tread+treadp,tcopy,treadk,
            tconv,tstore+tstorep,tend-tstart,tsending,treceiving,tparallel);
    }
    MPI_Finalize();
    return 0;
}
