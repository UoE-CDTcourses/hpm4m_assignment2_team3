#include <iostream>
#include <cmath>
#include <mpi.h>

using namespace std;

static const double PI = 3.1415926536; 

int main(int argc, char* argv[]){

int rank, size, ierr; // MPI related variables
MPI_Comm comm;
comm  = MPI_COMM_WORLD;

MPI_Init(NULL,NULL);
MPI_Comm_rank(comm, &rank);            
MPI_Comm_size(comm, &size);

MPI_Barrier(comm);
// Start timing the code
double t_start = MPI_Wtime();

int N_size = 10; // Number of processes
int root = 0; // Define the root proccess
int M = 2202;//100;  // M length intervals
int N = pow(10,6); // N time intervals
double T = atof(argv[1]);  // get final time from input argument
double U[2][M+1];  // stores the numerical values of function U; two rows to also store values of previous time step 
double U_mpi[2][M+1]; // stores numerical values of function U solved with MPI
double Usol[M+1];  // stores true solution 
double dt = T/N;
double dx = 1./M;
double dtdx = dt/(dx*dx);
int J = 2 + ((M-2) / N_size); // Number of points to be given to each process
if(rank == root)
{
cout<< "\ndx="<<dx<<", dt="<<dt<<", dt/dx²="<< dtdx<< ", J=" << J << endl;
}
// initialize numerical array with given conditions for serial and MPI
U[0][0]=0, U[0][M]=0, U[1][0]=0, U[1][M]=0;
U_mpi[0][0]=0, U_mpi[0][M]=0, U_mpi[1][0]=0, U_mpi[1][M]=0;
for(int m=1; m<M; ++m){
	U[0][m] = sin(2*PI*m*dx) + 2*sin(5*PI*m*dx) + 3*sin(20*PI*m*dx);
	U_mpi[0][m] = U[0][m];
}
if (rank == root)  //the root process only sends and receives from the end of the interval
{
	for(int i=1; i<=N; ++i){
	for (int m=1; m<J-1; ++m){
		U_mpi[1][m] = U_mpi[0][m] + dtdx * (U_mpi[0][m-1] - 2*U_mpi[0][m] + U_mpi[0][m+1]);	
	}
	// Send and receive the value of the PDE at the border of the interval
	MPI_Send(&U_mpi[1][J-2], 1, MPI_DOUBLE, 1, 0, comm);
	MPI_Recv(&U_mpi[1][J-1], 1, MPI_DOUBLE, 1, 0, comm,  MPI_STATUS_IGNORE);
	// update "old" values	
	for(int m=1; m<J; ++m){
		U_mpi[0][m] = U_mpi[1][m];
	}
}
}
if (rank == N_size - 1) //the final process only sends and receives from the start of the interval
{
	for(int i = 1; i<= N; ++i){
	for(int m= (J-1) + (rank -1)* (J-2); m<M; ++m){
		U_mpi[1][m] = U_mpi[0][m] + dtdx * (U_mpi[0][m-1] - 2*U_mpi[0][m] + U_mpi[0][m+1]);	
	}
	// Send and receive the value of the PDE at the border of the interval
	MPI_Send(&U_mpi[1][(J-1) + (rank -1)* (J-2)] , 1, MPI_DOUBLE, rank - 1 , 0, comm);
	MPI_Recv(&U_mpi[1][(J-1) + (rank -1)* (J-2)- 1], 1, MPI_DOUBLE, rank - 1, 0, comm,  MPI_STATUS_IGNORE);
	// update "old" values	
	for(int m= (J-1) + (rank -1)* (J-2) - 1; m<M; ++m){
		U_mpi[0][m] = U_mpi[1][m];
	}
}
}
if(rank!= N_size - 1 and rank != root) //all other process send and receive from thestart and end of the interval
{
	for(int i = 1; i<= N; ++i){

	for(int m = (J-1) + (rank -1)* (J-2); m<(J-1) + (rank) * (J-2); ++m){
		U_mpi[1][m] = U_mpi[0][m] + dtdx * (U_mpi[0][m-1] - 2*U_mpi[0][m] + U_mpi[0][m+1]);	
	}
	// Send and receive the value of the PDE at the border of the interval
	MPI_Send(&U_mpi[1][(J-1) + (rank -1)* (J-2)], 1, MPI_DOUBLE, rank - 1 , 0, comm);
	MPI_Send(&U_mpi[1][(J-1) + (rank) * (J-2)-1], 1, MPI_DOUBLE, rank + 1 , 0, comm);
	MPI_Recv(&U_mpi[1][(J-1) + (rank -1)* (J-2)-1], 1, MPI_DOUBLE, rank - 1, 0, comm,  MPI_STATUS_IGNORE);
	MPI_Recv(&U_mpi[1][(J-1) + (rank) * (J-2) ], 1, MPI_DOUBLE, rank + 1, 0, comm,  MPI_STATUS_IGNORE);
	// update "old" values	
	for(int m=(J-1) + (rank -1)* (J-2) - 1; m<=(J-1) + (rank) * (J-2); ++m){
		U_mpi[0][m] = U_mpi[1][m];
	}
}
}

//Ensure all processes have reached this point
MPI_Barrier(comm);

double U_pass[J-2]; //stores the U values to be passed to the root process..
//.. indices different for the root process compared to all other processes.
if(rank == 0)
{
	for (size_t i = 0; i < J - 2 ; i++)
	{
		U_pass[i] = U_mpi[1][i+2];
	}
	
}

if(rank != 0)
{
	for (size_t i = 0; i < J - 2; i++)
	{
		U_pass[i] = U_mpi[1][(J-1) + (rank -1)* (J-2) +1+ i];
	}
	
}

double U_mpians[2][M+1]; // stores the final answer to be printed in the root process
// Gathers U_pass from all the processes in the root process
MPI_Gather(U_pass, J-2, MPI_DOUBLE, U_mpians[0], J-2, MPI_DOUBLE, 0, comm);

// Ensures all of the values of U go in the right order...
for (int i = 0; i <= M; i++)
{
	U_mpians[1][i+2]=U_mpians[0][i];
}
// ... with the first two U values added manually.
if(rank == 0)
{
U_mpians[1][0]=U_mpi[1][0];
U_mpians[1][1]=U_mpi[1][1];
}

// use numerical scheme to obtain the future values of U on the M+1 space points
for(int i=1; i<=N; ++i){
	for (int m=1; m<M; ++m){
		U[1][m] = U[0][m] + dtdx * (U[0][m-1] - 2*U[0][m] + U[0][m+1]);	
	}
	// update "old" values	
	for(int m=1; m<M; ++m){
		U[0][m] = U[1][m];
	}
}

if(rank == 0)
{
	// print out array entries of numerical (both serial and parallel) solution next to true solution
	cout << "\nTrue and numerical values (both serial and MPI) at M="<<M<<" space points at time T="<<T<<":"<<endl;
	cout << "\nTrue values           Numerical solutions serial           Numerical solutions MPI\n"<<endl;
	for(int m=0; m<=M; ++m){
		Usol[m] = exp(-4*PI*PI*T)*sin(2*PI*m*dx) + 2*exp(-25*PI*PI*T)*sin(5*PI*m*dx) + 3*exp(-400*PI*PI*T)*sin(20*PI*M*dx);		
		cout << Usol[m] << "            " << U[1][m] << "\t\t\t\t"<< U_mpians[1][m]<< endl;
		// note that we did not really need to store the true solution in the array just to print out the values.
	}
}
//Ensure all processes have reached this point
MPI_Barrier(comm);
// Record time taken for the code to run
double t_end = MPI_Wtime();
double run_time = t_end - t_start;

// The root process returns the time total taken for the computation
if (rank == root)
{
	cout<< "\nThe code took  "<< run_time << " seconds to run.\n";
}

MPI_Finalize();
}
