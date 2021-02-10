#include <iostream>
#include <cmath>
#include <mpi.h>
#include <fstream>
using namespace std;

//Used this https://www.archer2.ac.uk/training/courses/200514-mpi/

int main(){

   //Instantiating a file to write the exact solution
   //and the numerical solution to.
   //ofstream file;
   //file.open("amair_week4_solutions_with labels.txt");

   //Initiating rank and size variables.
   int rank;
   int size;

   //Setting the outer points of the 1D domain.
   constexpr double a = 0;
   constexpr double b = 1;

   //Setting the constant pi.
   constexpr double pi = 3.1415926536;

   //These constants have been chosen for when 6 processes.
   //are being used

   //Setting the number of intervals that the 1D domain is
   //divided into.
   constexpr int M = 505;

   //Setting the number of time steps to consider after the
   //the initial condition.
   constexpr int N = 100000;

   //Setting the final time.
   constexpr double T = 0.1;

   //Resultant number of grid points including boundaries.
   constexpr int G = M + 1;

   //Setting delta x.
   constexpr double dx = (b - a)/M;

   //Setting delta t.
   constexpr double dt = T/N;

   //Setting lambda value.
   constexpr double lambda = dt/pow(dx, 2);

   //Setting number of processors.
   constexpr int nproc = 6;


   //Starting timer.
   double t_start, t_end;

   t_start = MPI_Wtime();

   //Defining a communicator.
   MPI_Comm comm;
   comm = MPI_COMM_WORLD;

   //Initialising parallel.
   MPI_Init(NULL,NULL);
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);


   //Aborting code if nproc != size
   if(size != nproc) {
    cout<<"nproc = "<<nproc<<" does not fit number of processes requested which is "
	<<size<<". Restart with matching number!"<<endl;
    MPI_Abort(comm, 911);
    MPI_Finalize();
    return 0;
   }

   //Number of grid points assigned to each process(including overlaps).
   constexpr int J = (G - 2)/nproc + 2;


   //Initialising the array for the solution at the grid points
   //assigned to each process, one row for current time step
   //and one row for the previous time step.
   double Useg[2][J];

   //Initialising an array to store the final solution on the
   //interior grid points for each process.
   double UIseg[J - 2];

   //Defining a vector to store the final solution on the interior
   //grid points of the entire domain.
   double UI[M - 1];

   //Defining a vector to store the final solution including boundary
   //grid points.
   double U[M + 1];

   //Defining a vector to store the exact solution at the final time.
   double Uex[M + 1];

   //Filling the array for the entire solution with the initial condition.
   U[0] = 0;
   U[M] = 0;

   for (int m = 1; m < M; ++m){
      U[m] = sin(2*pi*m*dx) + 2*sin(5*pi*m*dx) + 3*sin(20*pi*m*dx);
   }

   //Initialising each segment solution vector
   for (int j = 0; j < J; ++j){
      Useg[0][j] = U[rank*(J - 2) + j];
   }

//********Advancing the forward Euler scheme************************************************************************************
//********and filling each processor's segment with the corresponding chunk of the numerical solution.**************************

   for (int n = 1; n < N + 1; ++n){

      //Special case of process 0.
      if (rank == 0){

         //Computing the numerical solution
         //on the interior nodes of the segment.
         for (int j = 1; j < J - 1; ++j){
            Useg[1][j] = (lambda*Useg[0][j - 1] + (1 - 2*lambda)*Useg[0][j]
                          + lambda*Useg[0][j + 1]);
         }

         //"Sending to the right."
         //Sending the nodal numerical solution at Useg[1, J - 2] on process 0's segment
         //to be the nodal numerical solution  at Useg[1, 0] on process 1's segment.
         MPI_Ssend(&Useg[1][J - 2], 1, MPI_DOUBLE, rank + 1, 1, comm);


         //"Receiving from the right."
         //Receiving the nodal numerical solution at Useg[1, 1] on process 1's segment
         //to be the nodal numerical solution at Useg[1, J - 1] on process 0's segment.
         MPI_Recv(&Useg[1][J - 1], 1, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
      }

      //The other special case of the final process (the process with the number label: size - 1).
      else if (rank == size - 1){

         //Computing the numerical solution
         //on the interior nodes of the segment.
         for (int j = 1; j < J - 1; ++j){
            Useg[1][j] = (lambda*Useg[0][j - 1] + (1 - 2*lambda)*Useg[0][j]
                          + lambda*Useg[0][j + 1]);
         }

         //"Receiving from the left".
         //Receiving the nodal numerical solution at Useg[1, J - 2] on the segment of the penultimate
         //process to be the numerical solution at Useg[1, 0] on the final process.
         MPI_Recv(&Useg[1][0], 1, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

         //"Sending to the left".
         //Sending the nodal numerical solution at Useg[1, 1] on the segment of the final process
         //to be the nodal numerical solution at Useg[1, J - 1] on the segment of the penultimate
         //process.
         MPI_Ssend(&Useg[1][1], 1, MPI_DOUBLE, rank - 1, 0, comm);

      }

      //The general case for all the other processes (neither 0 nor size - 1).
      else {

         //Computing the numerical solution
         //on the interior nodes of the segment.
         for (int j = 1; j < J - 1; ++j){
            Useg[1][j] = (lambda*Useg[0][j - 1] + (1 - 2*lambda)*Useg[0][j]
                          + lambda*Useg[0][j + 1]);
         }

         //"Receiving from the left".
         //Receiving the numerical solution at Useg[1, J - 2] on the segment of process rank - 1
         //to be the numerical solution at Useg[1, 0] on the segment of this process rank.
         MPI_Recv(&Useg[1][0], 1, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

         //"Sending to the right".
         //Sending the numerical solution at Useg[1, J - 2] on the segment of this process rank
         //to be the numerical solution at Useg[1, 0] on the segment of process rank + 1.
         MPI_Ssend(&Useg[1][J - 2], 1, MPI_DOUBLE, rank + 1, 1, comm);

         //"Receiving from the right".
         //Receiving the numerical solution at Useg[1, 1] on the segment of process rank + 1
         //to be the numerical solution at Useg[1, J -1] on the segment of this process rank.
         MPI_Recv(&Useg[1][J - 1], 1, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);

         //"Sending to the left".
         //Sending the numerical solution at Useg[1 ,1] on the segment of this process rank
         //to be the numerical solution at Useg[1, J - 1] on the segment of process
         //rank - 1.
         MPI_Ssend(&Useg[1][1], 1, MPI_DOUBLE, rank - 1, 0, comm);

      }

      MPI_Barrier(comm);

      //Updating old values on each process's segment.
      for (int j = 0; j < J; ++j){
         Useg[0][j] = Useg[1][j];
      }

      //Waiting until all processes have computed then sent/received,
      //before moving onto the next iteration of the time loop.
      MPI_Barrier(comm);

   }

//********Gathering each segment's solution on rank 0.**************************************************************************

   //Storing the numerical solution values from the interior nodes of each process.
   for (int i = 0; i < J - 2; ++i){
      UIseg[i] = Useg[1][i + 1];
   }

   //Waiting until the interior node values have been collected for each process.
   MPI_Barrier(comm);

   //Gathering all interior node values for each segment in order of process, onto UI on process 0.
   MPI_Gather(&UIseg, J - 2, MPI_DOUBLE, UI, J - 2, MPI_DOUBLE, 0, comm);

   //Waiting until gather has been fully completed.
   MPI_Barrier(comm);

   //Entering the gathered interior values into the full solution vector on process 0.
   if (rank == 0){
      for (int m = 1; m < M; ++m){
         U[m] = UI[m - 1];
      }

      //Updating the exact solution at the final time.
      Uex[0] = 0;
      Uex[M] = 0;
      for (int m = 1; m < M; ++m){
         Uex[m] = (exp(-4*pow(pi, 2)*T)*sin(2*pi*m*dx)
                   + 2*exp(-25*pow(pi, 2)*T)*sin(5*pi*m*dx)
                   + 3*exp(-400*pow(pi, 2)*T)*sin(20*pi*m*dx));
      }

      //Printing the exact and numerical solutions at the final time.
      cout << "Numerical" << "          " << "  exact  " << endl;
      //file << "Numerical" << "          " << "  exact  " << endl;
      for (int m = 0; m < M + 1; ++m){
         cout << U[m] << "                   " << Uex[m] << endl;
         //file << U[m] << "                   " << Uex[m] << endl;
      }

      t_end = MPI_Wtime();

      cout << "Total running time: " << t_end - t_start << endl;
      //file << "Total running time: " << t_end - t_start << endl;
   }

   //Ending parallel.
   MPI_Finalize();


   return 0;
}
