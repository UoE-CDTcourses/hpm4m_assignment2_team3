#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
using namespace std;

static const double pi = 3.1415926536;

int main(int argc, char* argv[]){

   int rank, nproc, ierr;
   double T = atof(argv[1]);
   int M = stoi(argv[2]);
   int N = stoi(argv[3]);
    
   double dt = (double)T/(double)N;
   double dx = 1/(double)(M-1);
   double courant = dt/(dx*dx);

   MPI_Comm comm;
   MPI_Init(NULL,NULL);
   // Initialising variables to be used during the halo swapping procedure.
   int rank1,rank2;
   double received_value;
   double value_to_send;
   double x[M];
   int m;

   // Variable to gather the final answer into.
   double U_final[M];

   MPI_Comm_size(MPI_COMM_WORLD, &nproc);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);

   int J = 2+(M-2)/nproc;
   double U[2][J];
//Setting the initial conditions in all blocks, taking special note of the processes containing the boundary cells.
   for(int i = 0; i<nproc; i++){
      if(rank==i){
         for(int j = 0;j<J;j++ ){
            m = (rank*(J-2))+j;
            U[0][j] = sin(2*pi*m*dx) + 2*sin(5*pi*m*dx) + 3*sin(20*pi*m*dx);
         }
      }
      if(rank == 0){U[0][0] = 0;}
      if(rank == nproc-1){U[0][J-1] = 0;}
   }


for(int i = 1;i<=N; i++){
// Update middle values
   for(int j=1;j<J-1;j++){
      U[1][j] = U[0][j] + courant * (U[0][j-1] - 2*U[0][j] + U[0][j+1]);
   }

// Loop through the number of overlaps and perform the halo swap using send/receive commands for each pair of values to be swapped 
   for(int i = 0;i<nproc-1;i++){
      rank1 = i;
      rank2 = i+1;

      if(rank == rank1){
         value_to_send = U[1][J-2];
         MPI_Ssend(&value_to_send,1,MPI_DOUBLE,rank2,0,MPI_COMM_WORLD);
      }

      if(rank == rank2){
         MPI_Recv(&received_value,1,MPI_DOUBLE,rank1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         U[1][0] = received_value;
      }

      if(rank == rank2){
         value_to_send = U[1][1];
         MPI_Ssend(&value_to_send,1,MPI_DOUBLE,rank1,1,MPI_COMM_WORLD);
      }

      if(rank == rank1){
         MPI_Recv(&received_value,1,MPI_DOUBLE,rank2,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         U[1][J-1] = received_value;
      }
   }
   // Take care of the boundary values.
   if(rank == 0){ U[1][0] = 0;}
   if(rank == nproc-1){ U[1][J-1] = 0;}

   //Update the previous time step.
   for(int j=0;j<J;j++){
      U[0][j] = U[1][j];
   }
}

// Calculate the spatial grid
for(int m=0;m<M;m++){
   x[m] = m*dx;
}

// Read out the results to a file. I realise now that this is overly complicated, there's no need to loop over the processes i don't think. This is because I am reading out pairs of (U,x) so their order doesn't matter for plotting them. The
// current method here means that the entries on the file are in the correct order going from x=0 to x=1. but clearly this is unneccesary.

if(rank==0){cout<<"U,x"<<endl;} //Print header
for(int r=0;r<nproc;r++){
   if(rank == r){
      for(int i = 0;i<J-2;i++){
         m = (rank*(J-2))+i;
         cout<< U[1][i]<<','<<x[m]<<endl; //prints the U,x pair to an individual line.
      }
   }
MPI_Barrier(MPI_COMM_WORLD);
}
//The overlapping regions leave two elements at the end to me manually read out.
if(rank==nproc-1){
   cout<<U[1][J-2]<<','<<x[M-2]<<endl;
   cout<<U[1][J-1]<<','<<x[M-1]<<endl;
}
MPI_Finalize();
}
