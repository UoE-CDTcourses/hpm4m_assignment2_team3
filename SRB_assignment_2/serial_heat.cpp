#include<cmath>
#include<iostream>

using namespace std;

static const double pi = 3.1415926536;

int main(int argc, char* argv[]){

    double T = atof(argv[1]);
    int M = stoi(argv[2]);
    int N = stoi(argv[3]);
    
    double U[2][M];
    double U_true[M];
    double dt = (double)T/(double)N;
    double dx = 1/(double)(M-1);
    double x[M];

    double courant = dt/(dx*dx);

    if(courant > 0.5){
        cout << "Scheme is unstable for selected parameters! Please ensure that dt/dx^2 <0.5."<< endl;
        return 0;
    }

    U[0][0] = 0; //Dirichlet BCs
    U[0][M-1] = 0;
    U[1][0] = 0;
    U[1][M-1] = 0;

    // Setting the functional initial condition.
    for(int m=1; m<M-1; m++){
        U[0][m] = sin(2*pi*m*dx) + 2*sin(5*pi*m*dx) + 3*sin(20*pi*m*dx);
    }

    // Run through time steps
    for(int i = 1;i<=N; i++){
        for(int m=1;m<M-1;m++){
            U[1][m] = U[0][m] + courant * (U[0][m-1] - 2*U[0][m] + U[0][m+1]);
        }

        for(int m=1;m<M-1;m++){
            U[0][m] = U[1][m];
        }
    }

// Constructing the array of spatial positions.

    for(int m=0;m<M;m++){
        x[m] = m*dx;
    }

// Outputting final results to the text file in a readable format, results in a text file with two rows. Top row is the spatial position and the bottom row is the value of U..
    for(int m=0;m<M-1;m++){
        cout<<x[m]<<',';
    }
    cout << x[M-1]<<endl;
    
    for(int m=0;m<M-1;m++){
        cout<<U[1][m]<<',';
    }
    cout << U[1][M-1]<<endl;

    // Outputting parameters as reduncancy and if simulations for this set of parameters wanted to be revisited.
    cout << T << ',' << M <<',' << N << endl;

    return 0;
}