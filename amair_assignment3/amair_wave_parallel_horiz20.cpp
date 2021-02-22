#include <iostream>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <iomanip>
using namespace std;

int main(){

   //Starting timer.
   double t_start, t_end;

   t_start = MPI_Wtime();

   //Initiating rank and size variables.
   int rank;
   int size;

   //Setting the corner points of the square domain.
   constexpr double x0 = -1;
   constexpr double xI = 1;
   constexpr double y0 = -1;
   constexpr double Y1 = 1;

   //Number of intervals on one strip (vertical or horizontal)of the square domain.
   constexpr int M = 2301;

   //Number of time intervals.
   constexpr int N = 5*M;

   //Setting the final time.
   constexpr double T = 1;

   //Number of horizontal/vertical strips on the domain, including boundary strips.
   constexpr int G = M + 1;

   //Setting delta x/y.
   constexpr double dxy = (xI - x0)/M;

   //Setting delta t .
   constexpr double dt = T/N;

   //Setting the time iterations at which to take snapshots
   constexpr int n1 = 0.333*N;

   constexpr int n2 = 0.666*N;

   constexpr int n3 = N;

   //Setting lambda = (delta t)/(delta x/y).
   constexpr double lambda = dt/dxy;

   //Setting number of processes.
   constexpr int nproc = 20;

   //Defining the MPI_Status.
   MPI_Status status;

   //Defining a communicator.
   MPI_Comm comm;
   comm = MPI_COMM_WORLD;

   //Initialising parallel.
   MPI_Init(NULL,NULL);
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   //Aborting code if nproc != size
   if (size != nproc) {
      cout<<"nproc = "<<nproc<<" does not fit number of processes requested which is "
      <<size<<". Restart with matching number!"<<endl;
      MPI_Abort(comm, 911);
      MPI_Finalize();
      return 0;
   }

   //Number of strips  of changing x and constant y assigned to each process.
   constexpr int J = (G - 2)/nproc + 2;

   //Dynamically defining the array for the full solution U[x][y].
   double** U = new double* [M + 1];

   for (int i = 0; i < M + 1; ++i){
      U[i] = new double [M + 1];
   }

   //Setting the outer boundary nodes of the full solution array to zero.
   for (int i = 0; i < M + 1; ++i){
      U[0][i] = 0;
      U[M][i] = 0;
      U[i][0] = 0;
      U[i][M] = 0;
   }

   //Setting the initial condition on the full solution array.
   for (int i = 1; i < M; ++i) {
      for (int j = 1; j < M; ++j) {
         U[i][j] = exp(-40 * (pow(x0 + i * dxy - 0.4, 2) + pow(y0 + j * dxy, 2)));
      }
   }

   //Printing initial condition to file.

   //Initiating a file.
   ofstream out {"U_t0.csv"};

   //Setting the precision of entries in the file to 4 significant figures.
   out << fixed << setprecision(4);

   //Saving the initial u(x, y) values on the array.
   for (int i = 0; i < M + 1; ++i) {
      for (int j = 0; j < M + 1; ++j) {
         out << U[i][j] << " ";
      }
      out << endl;
   }

   //Closing the csv file.
   out.close();

   //Dynamically allocating a 3 dimensional array to house the chunk of the
   //solution array that has been assigned to the process in question.

   //Creating space for a 2D array at each required time step.
   double*** Uchunk = new double** [3];

   //Creating space for the x dimension at each time step.
   for (int  t = 0; t < 3; ++t) {
      Uchunk[t] = new double* [M + 1];

      //Creating an array for the y dimension at each node in the x dimension.
      for (int i = 0; i < M + 1; ++i) {
         Uchunk[t][i] = new double [J];
      }
   }

   //Setting the boundary conditions relevant to the chunk of each rank.
   if (rank == 0){

      //At each time, setting the column which corresponds to y = -1, to 0,
      //and setting the entries corresponding to x = -1 and x = 1, to 0.
      for (int t = 0; t < 3; ++t) {
         for (int i = 0; i < M + 1; ++i){
            Uchunk[t][i][0] = 0;
         }
         for (int j = 0; j < J; ++j){
            Uchunk[t][0][j] = 0;
            Uchunk[t][M][j] = 0;
         }
      }
   }

   else if (rank == nproc - 1){

      //At each time, setting the column which corresponds to y = 1, to 0,
      //and setting the entries corresponding to x = -1 and x = 1, to 0.
      for (int t = 0; t < 3; ++t) {
         for (int i = 0; i < M + 1; ++i){
            Uchunk[t][i][J - 1] = 0;
         }
         for (int j = 0; j < J; ++j){
            Uchunk[t][0][j] = 0;
            Uchunk[t][M][j] = 0;
         }
      }
   }

   else{

      //At each time,
      //setting the entries corresponding to x = -1 and x = 1, to 0.
      for (int n = 0; n < 3; ++n) {
         for (int j = 0; j < J; ++j){
            Uchunk[n][0][j] = 0;
            Uchunk[n][M][j] = 0;
         }
       }
    }

   //Setting the initial condition and condition at t = dt on each process's
   //chunk.
   for (int n = 0; n < 2; ++n){
      for (int i = 0; i < M + 1; ++i){
         for (int j = 0; j < J; ++j){
         Uchunk[n][i][j] = U[i][rank*(J - 2) + j];
         }
      }
   }

   //Dynamically allocating a 1D array for sending Uchunk[i][1] on current rank to
   //Uchunk[i][J - 1] on rank - 1. Only sending the interior points hence M - 1.
   double* Usendm = new double [M - 1];

   //Dynamically allocating a 1D array for sending Uchunk[i][J - 2] on current rank to
   //Uchunk[i][0] on rank + 1. Only sending the interior points hence M - 1.
   double* Usendp = new double [M - 1];

   //Dynamically allocating a 1D array for receiving Uchunk[i][J - 2] from rank - 1
   //to be Uchunk[i][0] on current rank. Only sending the interior points hence M-1.
   double* Urecvm = new double [M - 1];

   //Dynamically allocating a 1D array for receiving Uchunk[i][1] from rank + 1
   //to be Uchunk[i][J - 1] on current rank. Only sending the interior points hence M-1.
   double* Urecvp = new double [M - 1];

   //Dynamically allocating an M - 1 entry array to send the interior solution on a
   //strip that belongs to this process, to the process with rank 0.
   double* Usends = new double [M - 1];

   //Dynamically allocating an M - 1 entry array to receive interior solutions on
   //strips that belong to processes with rank not equal to 0.
   double* Urecvs = new double [M - 1];

   //********Advancing the numerical scheme in time.*************************

   for (int n = 0; n < N + 1; ++n) {
      //Special case of process 0.
      if (rank == 0) {
         //Compute numerical solution.
         for (int i = 1; i < M; ++i){
            for (int j = 1; j < J - 1; ++j){
               Uchunk[2][i][j] = (pow(lambda, 2)*(Uchunk[1][i - 1][j] + Uchunk[1][i][j - 1] - 4*Uchunk[1][i][j] + Uchunk[1][i + 1][j] + Uchunk[1][i][j + 1])
                                  + 2*Uchunk[1][i][j] - Uchunk[0][i][j]);
            }
         }

         //Fill Usendp with Uchunk[i][J - 2] to send to rank 1 to be Uchunk[i][0].
         for (int i = 0; i < M - 1; ++i){
            Usendp[i] = Uchunk[2][i + 1][J - 2];
         }

         //Sending Usendp to process with rank 1 to be Uchunk[i][0].
         MPI_Ssend(Usendp, M - 1, MPI_DOUBLE, rank + 1, 1, comm);

         //Receiving Usendm from process with rank 1 to be Uchunk[i][J - 1].
         MPI_Recv(Urecvp, M - 1, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);

         //Unpacking vector received from rank 1 into Uchunk[i][J - 1].
         for (int i = 0; i < M - 1; ++ i){
            Uchunk[2][i + 1][J - 1] = Urecvp[i];
         }

      }

      //Special case of process nproc - 1.
      else if (rank == nproc - 1){
         //Compute numerical solution.
         for (int i = 1; i < M; ++i){
            for (int j = 1; j < J - 1; ++j){
               Uchunk[2][i][j] = (pow(lambda, 2)*(Uchunk[1][i - 1][j] + Uchunk[1][i][j - 1] - 4*Uchunk[1][i][j] + Uchunk[1][i + 1][j] + Uchunk[1][i][j + 1])
                                  + 2*Uchunk[1][i][j] - Uchunk[0][i][j]);
            }
         }

         //Fill Usendm with Uchunk[i][1] to send to rank nproc - 2 to be Uchunk[i][J - 1]
         //on process nproc -2.
         for (int i = 0; i < M - 1; ++i){
            Usendm[i] = Uchunk[2][i + 1][1];
         }

         //Receiving Usendp from process with rank nproc - 2 to be Uchunk[i][0].
         MPI_Recv(Urecvm, M - 1, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

         //Sending Usendm to process with rank nproc - 2 to be Uchunk[i][J - 1].
         MPI_Ssend(Usendm, M - 1, MPI_DOUBLE, rank - 1, 0, comm);

         //Unpacking vector received from rank nproc - 2 into Uchunk[i][0].
         for (int i = 0; i < M - 1; ++ i){
            Uchunk[2][i + 1][0] = Urecvm[i];
         }
      }

      //All intermediary processes.
      else{
         //Compute numerical solution.
         for (int i = 1; i < M; ++i){
            for (int j = 1; j < J - 1; ++j){
               Uchunk[2][i][j] = (pow(lambda, 2)*(Uchunk[1][i - 1][j] + Uchunk[1][i][j - 1] - 4*Uchunk[1][i][j] + Uchunk[1][i + 1][j] + Uchunk[1][i][j + 1])
                                  + 2*Uchunk[1][i][j] - Uchunk[0][i][j]);
            }
         }

         //Fill Usendm with Uchunk[i][1] to send to rank -1 to be Uchunk[i][J - 1].
         for (int i = 0; i < M - 1; ++i){
            Usendm[i] = Uchunk[2][i + 1][1];
         }

         //Fill Usendp with Uchunk[i][J - 2] to send to rank +1 to be Uchunk[i][0].
         for (int i = 0; i < M - 1; ++i){
            Usendp[i] = Uchunk[2][i + 1][J - 2];
         }

         //Receiving Usendp from process with rank -1 to be Uchunk[i][0].
         MPI_Recv(Urecvm, M - 1, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

         //Sending Usendp to process with rank +1 to be Uchunk[i][0].
         MPI_Ssend(Usendp, M - 1, MPI_DOUBLE, rank + 1, 1, comm);

         //Receiving Usendm from process with rank +1 to be Uchunk[i][J - 1].
         MPI_Recv(Urecvp, M - 1, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);

         //Sending Usendm to process with rank -1 to be Uchunk[i][J - 1].
         MPI_Ssend(Usendm, M - 1, MPI_DOUBLE, rank - 1, 0, comm);

         //Unpacking Urecvm into Uchunk[i][0].
         for (int i = 0; i < M - 1; ++ i){
            Uchunk[2][i + 1][0] = Urecvm[i];
         }

         //Unpacking Urecvp into Uchunk[i][J - 1].
         for (int i = 0; i < M - 1; ++ i){
            Uchunk[2][i + 1][J - 1] = Urecvp[i];
         }

      }

      MPI_Barrier(comm);

      //Updating old values on each chunk
      for (int i = 0; i < M + 1; ++i){
         for (int j = 0; j < J; ++j){
            Uchunk[0][i][j] = Uchunk[1][i][j];
            Uchunk[1][i][j] = Uchunk[2][i][j];
         }
      }

      MPI_Barrier(comm);

      //Printing out the files for given fixed times
	  //This syntax translates as if t = t1 or t = t2 or t = t3.
	  if(n==n1 || n==n2 || n==n3){

	     if(rank != 0){
            //Sending the strip at each interior fixed value of y.
            for (int j = 1; j < J - 1; ++j){
                for (int i = 0; i < M - 1; ++i){
                   Usends[i] = Uchunk[2][i + 1][j];
                }
                MPI_Ssend(Usends, M - 1, MPI_DOUBLE, 0, j, comm);
            }
	     }

	     else if(rank == 0){
            //Receiving each strip of interior fixed values of y.
            for (int c = 0; c < (nproc - 1)*(J - 2); ++c){
               MPI_Recv(Urecvs, M - 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);

               //Storing source and tag of the receipt.
               int source = status.MPI_SOURCE;
               int tag = status.MPI_TAG;

               //Unpacking the strip of values into their correct location.
               for (int i = 1; i < M; ++i){
                  U[i][(J - 2)*source + tag] = Urecvs[i - 1];
               }
            }

            //Filling columns 1 to J - 1 in the solution array
            for (int j = 1; j < J - 1; ++j){
               for (int i = 1; i < M; ++ i){
                  U[i][j] = Uchunk[2][i][j];
               }
            }

            //Creating a label to be appended to the file name to distinguish
		    //files with t1, t2 and t3 data.
		    stringstream ss;
            ss << fixed << setprecision(2) << n*dt; // this ensures that the double value gets converted
		    string time = ss.str();						 // to string with only 2 trailing digits.

		    //Naming the file based on what time is being considered.
		    ofstream out {"U_t"+ss.str()+".csv"};
            //Setting the solution output at each node to be to 4 significant figures.
		    out << fixed << setprecision(4);
		    //Saving the solution at the given time to its corresponding file.
			for(int i=0; i < M + 1; ++i){
			   for(int j=0; j< M + 1; ++j){
					out<<U[i][j]<<" ";
				}
				out<<endl;
			}
		    out.close();
	     }
	  }
   }

   //Deallocating memory

   //Deallocating the solution matrix.
   for (int i = 0; i < M + 1; ++i){
         delete[] U[i];
   }

   delete[] U;

   //Deallocating the Uchunk matrices.
   for (int n = 0; n < 3; ++n){
      for (int i = 0; i < M + 1; ++i){
         delete[] Uchunk[n][i];
      }
      delete[] Uchunk[n];
   }

   delete[] Uchunk;

   //Deallocating the 1D arrays.
   delete [] Usendm;
   delete [] Usendp;
   delete [] Urecvm;
   delete [] Urecvp;
   delete [] Usends;
   delete [] Urecvs;

   //Ending parallel.
   MPI_Finalize();

   t_end = MPI_Wtime();

   cout << "Total running time: " << t_end - t_start << endl;

   return 0;
}
