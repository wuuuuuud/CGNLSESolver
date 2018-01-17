
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <ctime>
#include <cstring>
#include "windows.h"

#ifdef __CUDACC__
#define K2(block,thread) <<< block, thread>>>
#else
#define K2(block,thread)
#endif
#define square(x) (x*x)

const int TGridSize = (8192); // 2^16;
__constant__ double TInitialWidth = (65e-15);
#define TSpan (20e-12) //Tspan 20 ps
//#define DISPERSION_ONLY // No gamma
//#define NO_RAMAN
//#define ZERO_DISPERSION
//#define NO_DIFFERENTIAL
#define RECORD_HISTORY 0
#define USE_FILTER 0
const double PI = 3.14159265358979;
const double c = 3e8;
const int ZActualStep = 2048;
const double TStep = (TSpan / (TGridSize + 0.0));
const double inverseTStep = 1 / TStep;
const int ZCount = 2048;
const double ZStep = 1.93 / ZCount;
const cuDoubleComplex cZStep = { ZStep,0 };
const double centerLambda = 1060e-9;
const double omega0 = c / centerLambda;
const cuDoubleComplex one = { 1,0 };
const cuDoubleComplex cromega0i = { 0,1 / omega0 /2/PI}; // complex and reversed omega0
const cuDoubleComplex twothirds = { 2/3.0,0 };
const cuDoubleComplex onesixths = { 1/6.0,0 };
const cuDoubleComplex inverseTGridSize = { 1 / (0.0 + TGridSize),0 };
const double inverseT2GridSize =  1 / (0.0 + 2*TGridSize);
const double dhalf = 0.5; // 0.5 in double form;
double Amplitude = 80;
const double Angle0 = 0 / 180.0 * PI;
const double AngleStep = 1 / 180.0*PI;
const int AngleCount = 90;
const double gamma = 0.011;
const cuDoubleComplex cgammai = { 0,gamma };
const double rCP[13] = { 56.25,100,231.25,362.50,463,497,\
611.5,691.67,793.67,835.50,930,1080,1215 };
const double rPI[13] = { 1,11.4,36.67,67.67,74,4.5,6.8,4.6,4.2,4.5,2.7,3.1,3 };
const double rGF[13] = { 52.1,110.42,175,162.5,135.33,24.5,41.5,155,59.5,64.3,150,91,160 };
const double rLF[13] = { 17.37,38.81,58.33,54.17,45.11,8.17,13.83,51.67,19.83,21.43,50,30.33,53.33 };
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void GPUMemoryAllocate();
void GPUMemoryClear();
void NConstruct(cuDoubleComplex *u, cuDoubleComplex *v, double *r, cuDoubleComplex* dstu, cuDoubleComplex* dstv);
void PrintVairableD(double* gpumemory, int n, char* name);
void PrintVairableZ(cuDoubleComplex* gpumemory, int n, char* name);
void PrintVairableZA(cuDoubleComplex* gpumemory, int n, char* name);
template <class T>
void SaveVariableBinary(T* v, int count, char* filename);
template <class T>
void SaveGPUVariableBinary(T* GPUMemory, int count, char* filename)
{
	T* temp = (T*)malloc(count * sizeof(T));
	cudaMemcpy(temp, GPUMemory, count * sizeof(T), cudaMemcpyDeviceToHost);
	SaveVariableBinary(temp, count, filename);
	free(temp);
}
char foldername[100];
#pragma region GPU Memory declaration



// u: Envelope at fast axis
// v: Envelope at slow axis
// r: Raman response
cuDoubleComplex* u, *v, *u2, *v2, *u3, *v3, *fr,*u0;
//  u history
cuDoubleComplex* uh[ZActualStep];
cuDoubleComplex* ua[AngleCount];
cuDoubleComplex* va[AngleCount];
// temp variables
const int tempVariableCount = 6;
cuDoubleComplex* tv[tempVariableCount];
double* tvd[tempVariableCount];

cuDoubleComplex* k[4];
cuDoubleComplex* l[4];
// expand u,v,r for convolution, they are twice as large. 
double *cu, *cr, *cv,*r;
// FTed cu, cr, cv 
cuDoubleComplex *fcu, *fcr, *fcv;
// fu: FT of u
// fv: FT of v
cufftDoubleComplex* fu, *fv;
// t: Time grid
double* t;
// dispersion vector
cuDoubleComplex* dxvector;
cuDoubleComplex* dyvector;
#pragma endregion
// FFT plan
cufftHandle fftplan;
cufftHandle fftplan2,fftplan2I;
// cuBlas
cublasStatus_t stat;
cublasHandle_t handle;
// Aux functions

double abs(cufftDoubleComplex c)
{
	return sqrt(c.x*c.x + c.y*c.y);
}
#pragma region Kernels



__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void createGaussianPulseKernel(cuDoubleComplex* u, const double* t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < TGridSize)
	{
		u[i].x = exp(-t[i] * t[i] / (TInitialWidth*TInitialWidth));
		i += gridDim.x;
	}
}

__global__ void createTGridKernel(double* t, const double TStep)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < TGridSize)
	{
		t[i] = (i - TGridSize / 4)*TStep;
		i += gridDim.x;
	}
}

__global__ void vectorBoundResetKernel( cuDoubleComplex* u)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<TGridSize)
	{
		if (i < TGridSize/256 || i > (TGridSize -TGridSize/64))
		{
			u[i] = { 0,0 };
		}

	}
}

__global__ void vectorThresholdKernel(cuDoubleComplex* u)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<TGridSize)
	{
		if ((u[i].x*u[i].x+u[i].y*u[i].y)<1e-3)
		{
			u[i] = { 0,0 };
		}

	}
}
// v=v.u
// w=w.u
// u is constant
__global__ void vectorProductV3Kernel(const cuDoubleComplex* u, cuDoubleComplex *v, cuDoubleComplex* w)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double tempvx,tempwx;
	if (i < TGridSize *2)
	{
		tempvx = u[i].x*v[i].x - u[i].y*v[i].y;
		v[i].y = u[i].x*v[i].y + u[i].y*v[i].x;
		v[i].x = tempvx;

		tempwx = u[i].x*w[i].x - u[i].y*w[i].y;
		w[i].y = u[i].x*w[i].y + u[i].y*w[i].x;
		w[i].x = tempwx;
		//i += gridDim.x;
	}
}
// v= v.*u
__global__ void vectorProductKernel(const cuDoubleComplex* u, cuDoubleComplex *v)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double tempvx;
	if (i < TGridSize)
	{
		tempvx = u[i].x*v[i].x - u[i].y*v[i].y;
		v[i].y = u[i].x*v[i].y + u[i].y*v[i].x;
		v[i].x = tempvx;

		//i += gridDim.x;
	}
}

__global__ void vectorProduct2Kernel(cuDoubleComplex* u, cuDoubleComplex *v,cuDoubleComplex *w, cuDoubleComplex* dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	cuDoubleComplex temp;
	double tempx = 0;
	while (i < TGridSize)
	{

		temp.x = u[i].x*v[i].x - u[i].y*v[i].y;
		temp.y = u[i].x*v[i].y + u[i].y*v[i].x;
		tempx = temp.x*w[i].x - temp.y*w[i].y;
		dst[i].y = temp.x*w[i].y+temp.y*w[i].x;
		dst[i].x = tempx;
		i += gridDim.x;
	}
}

// dst1 = |u|.^2.*v
// dst2 = |u|.^2.*u
// dst3 = |v|.^2.*u
// dst4 = |v|.^2.*v
// dst5 = |u|.^2
// dst6 = |v|.^2
__global__ void vectorProductV1Kernel(cuDoubleComplex* u, cuDoubleComplex *v, \
	cuDoubleComplex* dst1,cuDoubleComplex* dst2, cuDoubleComplex* dst3, cuDoubleComplex* dst4,\
	double* dst5, double* dst6)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double tempux ,tempvx;
	cuDoubleComplex tempu,tempv;


	if (i < TGridSize)
	{
		tempu = u[i];
		tempv = v[i];
		tempux = tempu.x*tempu.x + tempu.y*tempu.y;
		tempvx = tempv.x*tempv.x + tempv.y*tempv.y;
		dst1[i].y = tempux*tempv.y;
		dst1[i].x = tempux*tempv.x;
		dst2[i].x = tempux*tempu.x;
		dst2[i].y = tempux*tempu.y;
		dst3[i].x = tempvx*tempu.x;
		dst3[i].y = tempvx*tempu.y;
		dst4[i].x = tempvx*tempv.x;
		dst4[i].y = tempvx*tempv.y;
		dst5[i] = tempux;
		dst6[i] = tempvx;
	//	i += gridDim.x;
	}
}
// dst = |u|^2
__global__ void vectorProductV0Kernel(cuDoubleComplex* u,  cuDoubleComplex* dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < TGridSize)
	{
		
		dst[i].x = (u[i].x*u[i].x + u[i].y*u[i].y);
		dst[i].y = 0;
		i += gridDim.x;
	}
}

// dst = a * src + b
__global__ void vectorLinearKernel(cuDoubleComplex* src, cuDoubleComplex* dst, cuDoubleComplex a, cuDoubleComplex b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double tempx;
	if (i < TGridSize)
	{
		tempx = src[i].x*a.x - src[i].y*a.y + b.x;
		dst[i].y= src[i].x*a.y + src[i].y*a.x + b.y;
		dst[i].x = tempx;
		//i += gridDim.x;
	}
}
// dst = a*u + v
// dst != u
// dst != v
__global__ void vectorLinearAddKernel(cuDoubleComplex* dst, cuDoubleComplex* u, cuDoubleComplex* v,const double a)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < TGridSize)
	{
		dst[i].y = u[i].y*a + v[i].y;
		dst[i].x = u[i].x*a + v[i].x;
		//i += gridDim.x;
	}
}

// v = a*u + v
__global__ void vectorLinearCummulativeAddKernel( cuDoubleComplex* u, cuDoubleComplex* v, const double a)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double tempx;
	if (i < TGridSize)
	{
		tempx = u[i].x*v[i].x - u[i].y*v[i].y;
		v[i].y = u[i].y*a + v[i].y;
		v[i].x = tempx;
		//i += gridDim.x;
	}
}
// Unsafe version
__global__ void vectorLinearKernelU23(cuDoubleComplex* src, cuDoubleComplex* dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < TGridSize)
	{
		dst[i] = { src[i].x*0.6666666666666666666666666, src[i].y*0.66666666666666666 };
		i += gridDim.x;
	}
}
// dst = (src[i+1]-src[i-1])/2
// dst[1]=dst[end]=0
// dst != src or error occurs
__global__ void vectorDifferentialKernel(cuDoubleComplex* src, cuDoubleComplex* dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double temp = 0;
	if (i != 0)
	{
		if (i < TGridSize - 1)
		{

			dst[i].x = (src[i + 1].x - src[i - 1].x) / 2;
			dst[i].y = (src[i + 1].y - src[i - 1].y) / 2;
			//i += gridDim.x;
		}
	}
	else
	{
		dst[0].x = dst[0].y = 0;
		dst[TGridSize].x = dst[TGridSize].y = 0;
	}
}

__global__ void vectorAddKernel(cuDoubleComplex* u, cuDoubleComplex *v, cuDoubleComplex *dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < TGridSize)
	{

		dst[i].y = u[i].y + v[i].y;
		dst[i].x = u[i].x + v[i].x;
		i += gridDim.x;
	}
}
// dst = u+v.*w
// dst != v or w
__global__ void vectorProductAddKernel(cuDoubleComplex* u, double *v,cuDoubleComplex *w, cuDoubleComplex *dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < TGridSize)
	{

		dst[i].y = u[i].y + v[i]*w[i].y;
		dst[i].x = u[i].x + v[i]*w[i].x;
		//i += gridDim.x;
	}
}

__global__ void FFTShiftCopyKernelD2Z(double *src, cuDoubleComplex *dst)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	double temp = 0;
	while (i < TGridSize)
	{

		dst[i].x = src[i + TGridSize];
		dst[i].y = 0;
		i += gridDim.x;
	}
}
// u= AI+1/6*k1+1/3*k2+1/3k3
__global__ void RungeKutta1Kernel(cuDoubleComplex* u, cuDoubleComplex *k1, cuDoubleComplex* k2, cuDoubleComplex* k3, cuDoubleComplex* AI)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < TGridSize)
	{
		u[i].x = AI[i].x + 0.16666666666666666667*k1[i].x + 0.33333333333333333333*k2[i].x + 0.33333333333333333333333333*k3[i].x;
		u[i].y = AI[i].y + 0.16666666666666666667*k1[i].y + 0.33333333333333333333*k2[i].y + 0.33333333333333333333333333*k3[i].y;
		//i += gridDim.x;
	}
}
// v = filtered(u);
__global__ void LinearFilterKernel(cuDoubleComplex* u,cuDoubleComplex* v)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int S = 15;
	if ((i < (TGridSize-S)) && (i>S-1))
	{
		v[i] = { 0,0 };
		for (int k=i-S;k<i+S+1;k++)
		{
			v[i].x += u[k].x;
			v[i].y += u[k].y;
		}
		v[i].x /= 2*S+1.0;
		v[i].y /= 2*S+1.0;
		//i += gridDim.x;
	}
	else if(i< TGridSize)
	{
		v[i] = { 0,0 };
	}
}
#pragma endregion
#pragma region CPU Generated



// Calculate temporal response of Raman according to Hollenbeck,2002
// First calculate at CPU, then cumemcpy to GPU memory at address r 
void createRamanKernel(double* r)
{
	double* hr = (double*)malloc(TGridSize * sizeof(double));
	int i = 0;
	while (i < TGridSize)
	{

		hr[i] = 0;

		for (int k = 0; k < 13; k++)
		{
			hr[i]+= rPI[k] * exp(-TStep*i*PI*c*rLF[k] * 1e2)*
				exp(-pow(PI*c*rGF[k] * 1e2*TStep, 2)*pow(i, 2) / 4)*sin(2 * PI*c*rCP[k] * 1e2*i*TStep);
			
		}
		//hr[i] *= 5.3e10;
		hr[i] *= 5.3e10;
		i++;
	}

	cudaMemcpy(r, hr, TGridSize * sizeof(double), cudaMemcpyHostToDevice);
#ifdef NO_RAMAN
	cudaMemset(r, 0, TGridSize * sizeof(double));
#endif // 

	free(hr);
}

// Calculate z transfer function in spectral domain.
// first calculate omega grid according to TStep
// then calculate dispersion transfer function with dispersion paramaters
// finally, copy it to GPU
void createDispersionKernel(cuDoubleComplex* dxkernel, cuDoubleComplex* dykernel)
{
	//double* ogrid = (double *)malloc(TGridSize * sizeof(double));
	double omax = 1 / (TStep);
	double ostep = omax / (TGridSize);
	cuDoubleComplex* tempkernel = (cuDoubleComplex*)malloc(TGridSize * sizeof(cuDoubleComplex));
	cuDoubleComplex* tempykernel = (cuDoubleComplex*)malloc(TGridSize * sizeof(cuDoubleComplex));
	auto ogrid = (double*)malloc(TGridSize * sizeof(double));
	double factorial[] = { 1,1,2,6,24,120,720 };
	//beta_2PCF = -2.93e-27;  % s ^ 2 / m  the PCF - 4.2e-27 30
	//beta_3PCF = 0.0717e-39; % s ^ 3 / m  the PCF  0.0681e-39 45
	//beta_4PCF = -5.936*2e-56; % 60
	//beta_5PCF = 6.896*6e-71; % 75
	//beta_6PCF = -4.137*24e-86; % 90
	const double beta[] = { 0,0,
							-2.93e-27,
							7.17e-41,
							-2 * 5.936e-56,
							6 * 6.896e-71,
							-24 * 4.137e-86 };
	//Lb = 6.3e-3; % ÅÄ³¤ m 6.3
	//beta_polarization = pi / Lb;
	//kesi = beta_polarization*wave0 / 2 / pi / c; % the inverse group velocity difference
	const double Lb = -6.3e-3;
	const double beta_polarization = PI / Lb;
	const double kesi = beta_polarization*centerLambda / 2.0 / PI / c;


	for (int j = 0; j < TGridSize; j++)
	{
		double o = (j < TGridSize / 2) ? (j*ostep) : ( (j - TGridSize )*ostep);
		o *= 2 * PI;
		ogrid[j] = o;
		//dispersion_PCF_x=exp((1i*omega_x.^2*deltaz*beta_2PCF/2+1i*omega_x.^3*deltaz*beta_3PCF/6+
		//1i*omega_x.^4*deltaz*beta_4PCF/24+1i*omega_x.^5*deltaz*beta_5PCF/120+1i*omega_x.^6*deltaz*beta_6PCF/720)/2);
		double _k = 0, _ky = 0;
		for (int jj = 2; jj < 7; jj++)
		{
			_k += pow(o, jj)*ZStep*beta[jj] / factorial[jj];
		}
		//polarization_x = exp(1i*(omega.*deltaz*kesi + deltaz*beta_polarization) / 2);
		//polarization_y = exp(-1i*(omega.*deltaz*kesi + deltaz*beta_polarization) / 2);
		_ky = _k;
		_k += o*ZStep*kesi + ZStep*beta_polarization;
		_k /= 2.0;
		_ky -= o*ZStep*kesi + ZStep*beta_polarization;
		_ky /= 2.0;
		tempkernel[j].x = cos(_k);
		tempkernel[j].y = sin(_k);
		tempykernel[j].x = cos(_ky);
		tempykernel[j].y = sin(_ky);
#ifdef ZERO_DISPERSION
		tempkernel[j].x = 1;
		tempkernel[j].y = 0;
		tempykernel[j].x = 1;
		tempykernel[j].y = 0;
#endif
	}
	cudaMemcpy(dxkernel, tempkernel, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dykernel, tempykernel, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);



	PrintVairableZ(dxkernel, TGridSize, "vectorx0.txt");
	SaveVariableBinary(ogrid, TGridSize, "oGrid.dbin");
	free(tempkernel);
	free(tempykernel);
	free(ogrid);
}
#pragma endregion

// d double
// i int
// z double complex
void saveSimulationEnvironments(char* foldername)
{
	SaveVariableBinary(&Angle0, 1, "angleStart.dbin");
	SaveVariableBinary(&AngleStep, 1, "angleStep.dbin");
	SaveVariableBinary(&AngleCount, 1, "angleCount.ibin");
	SaveVariableBinary(&Amplitude, 1, "amplitude.dbin");
	SaveVariableBinary(&TGridSize, 1, "tGridSize.ibin");
	SaveVariableBinary(&gamma, 1, "gamma.dbin");
	SaveVariableBinary(&TInitialWidth, 1, "t0.dbin");
	SaveVariableBinary(&ZStep,1,"zStep.dbin");
	SaveVariableBinary(&ZCount, 1, "zCount.ibin");
	SaveVariableBinary(rCP, 13, "rCP.dbin");
	SaveVariableBinary(rPI, 13, "rPI.dbin");
	SaveVariableBinary(rGF, 13, "rGF.dbin");
	SaveVariableBinary(rLF, 13, "rLF.dbin");
	SaveGPUVariableBinary(u0, TGridSize, "t0.zbin");
	SaveGPUVariableBinary(t, TGridSize, "tGrid.dbin");
	SaveGPUVariableBinary(dxvector, TGridSize, "dispersionU.zbin");
	SaveGPUVariableBinary(dyvector, TGridSize, "dispersionV.zbin");
}
void getTimeString(char* timestring)
{
	time_t now=time(0);
	struct tm* timeinfo;
	timeinfo = localtime(&now);
	sprintf(timestring, "%4d%02d%02d#%02d%02d%02d\0", timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
		timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
}
int main()
{
	fprintf(stderr, "Begin...\n");
	cuDoubleComplex* uu;
	double *tt;
	cudaError_t cudaStatus;
	stat = cublasCreate(&handle);
	uu = (cuDoubleComplex*)malloc(2*TGridSize * sizeof(cuDoubleComplex));
	tt = (double*)malloc(2*TGridSize * sizeof(double));
	cufftDoubleComplex* hfu = (cufftDoubleComplex*)malloc(TGridSize * sizeof(cufftDoubleComplex));
	getTimeString(foldername);
	CreateDirectory(foldername, NULL);
	SetCurrentDirectory(foldername);
	// GPU Initialize
	cudaSetDevice(0);
	// cufft Initialize
	cufftPlan1d(&fftplan, TGridSize, CUFFT_Z2Z, 1);
	cufftPlan1d(&fftplan2, 2 * TGridSize, CUFFT_D2Z, 1);
	cufftPlan1d(&fftplan2I, 2 * TGridSize, CUFFT_Z2D, 1);
	
	// Allocate GPU memory
	fprintf(stdout, "GPU opened.\n");
	GPUMemoryAllocate();
	fprintf(stderr, "Memory allocated.\n");
	
	// Create time grid
	createTGridKernel K2(4096, 64) (t, TStep);
	fprintf(stderr, "T calculated.\n");
	createGaussianPulseKernel K2(4096, 64) (u0, t);
	
	PrintVairableZ(u0,TGridSize, "u0.csv");
	
	// Raman response initialize
	createRamanKernel(r);
	cudaMemcpy(cr + TGridSize / 2, r, TGridSize * sizeof(double), cudaMemcpyDeviceToDevice);
	cufftExecD2Z(fftplan2, cr, fcr);
	createDispersionKernel(dxvector, dyvector);
	for (int a = 0; a<AngleCount; a++)
	{
		double Angle = Angle0 + a*AngleStep;
		printf("Angle: %e  ", Angle/PI*180.0);
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Initialize input pulse;
		vectorLinearKernel K2(4096, 64) (u0, v, { Amplitude * sin(Angle),0 }, { 0,0 });
		vectorLinearKernel K2(4096, 64) (u0, u, { Amplitude * cos(Angle),0 }, { 0,0 });
		for (int stepCount = 0; stepCount < ZActualStep; stepCount++)
		{
			cufftExecZ2Z(fftplan, u, fu, CUFFT_INVERSE);
			cufftExecZ2Z(fftplan, v, fv, CUFFT_INVERSE);

			vectorProductKernel K2(4096, 64) (dxvector, fu);
			vectorProductKernel K2(4096, 64) (dyvector, fv);

			// u3 === uI, v3 === vI
			cufftExecZ2Z(fftplan, fu, u3, CUFFT_FORWARD);
			cufftExecZ2Z(fftplan, fv, v3, CUFFT_FORWARD);

			cublasZscal(handle, TGridSize, &inverseTGridSize, u3, 1);
			cublasZscal(handle, TGridSize, &inverseTGridSize, v3, 1);
			// k1
			NConstruct(u, v, r, k[0], l[0]);
			// Linear operator
			cufftExecZ2Z(fftplan, k[0], fu, CUFFT_INVERSE);
			cufftExecZ2Z(fftplan, l[0], fv, CUFFT_INVERSE);

			vectorProductKernel K2(4096, 64) (dxvector, fu);
			vectorProductKernel K2(4096, 64) (dyvector, fv);

			cufftExecZ2Z(fftplan, fu, k[0], CUFFT_FORWARD);
			cufftExecZ2Z(fftplan, fv, l[0], CUFFT_FORWARD);

#if USE_FILTER
#ifndef NO_DIFFERENTIAL
			LinearFilterKernel K2(4096, 64) (k[0], tv[4]);
			LinearFilterKernel K2(4096, 64) (l[0], tv[5]);
			cublasZswap(handle, TGridSize, k[0], 1, tv[4], 1);
			cublasZswap(handle, TGridSize, l[0], 1, tv[5], 1);
#endif // !NO_DIFFERENTIAL
#endif



			cublasZscal(handle, TGridSize, &inverseTGridSize, k[0], 1);
			cublasZscal(handle, TGridSize, &inverseTGridSize, l[0], 1);
			//cublasZscal(handle, TGridSize, &cZStep, k[0], 1);
			//cublasZscal(handle, TGridSize, &cZStep, l[0], 1);
			// Runge-kutta step 2
			vectorLinearAddKernel K2(4096, 64)(u2, k[0], u3, dhalf);
			vectorLinearAddKernel K2(4096, 64)(v2, l[0], v3, dhalf);

			// k2
			NConstruct(u2, v2, r, k[1], l[1]);

			//cublasZscal(handle, TGridSize, &cZStep, k[1], 1);
			//cublasZscal(handle, TGridSize, &cZStep, l[1], 1);
			// Runge-kutta step 3
			vectorLinearAddKernel K2(4096, 64)(u2, k[1], u3, dhalf);
			vectorLinearAddKernel K2(4096, 64)(v2, l[1], v3, dhalf);


			// k3
			NConstruct(u2, v2, r, k[2], l[2]);

			//cublasZscal(handle, TGridSize, &cZStep, k[2], 1);
			//cublasZscal(handle, TGridSize, &cZStep, l[2], 1);
			// Runge-kutta step 4
			vectorLinearAddKernel K2(4096, 64)(u2, k[2], u3, 1);
			vectorLinearAddKernel K2(4096, 64)(v2, l[2], v3, 1);

			cufftExecZ2Z(fftplan, u2, fu, CUFFT_INVERSE);
			cufftExecZ2Z(fftplan, v2, fv, CUFFT_INVERSE);

			vectorProductKernel K2(4096, 64) (dxvector, fu);
			vectorProductKernel K2(4096, 64) (dyvector, fv);

			cufftExecZ2Z(fftplan, fu, u2, CUFFT_FORWARD);
			cufftExecZ2Z(fftplan, fv, v2, CUFFT_FORWARD);

			cublasZscal(handle, TGridSize, &inverseTGridSize, u2, 1);
			cublasZscal(handle, TGridSize, &inverseTGridSize, v2, 1);

			// k4
			NConstruct(u2, v2, r, k[3], l[3]);

			//cublasZscal(handle, TGridSize, &cZStep, k[3], 1);
			//cublasZscal(handle, TGridSize, &cZStep, l[3], 1);
			// Postprocess
			RungeKutta1Kernel K2(4096, 64) (u, k[0], k[1], k[2], u3);
			RungeKutta1Kernel K2(4096, 64) (v, l[0], l[1], l[2], v3);

			cufftExecZ2Z(fftplan, u, fu, CUFFT_INVERSE);
			cufftExecZ2Z(fftplan, v, fv, CUFFT_INVERSE);

			vectorProductKernel K2(4096, 64) (dxvector, fu);
			vectorProductKernel K2(4096, 64) (dyvector, fv);

			cufftExecZ2Z(fftplan, fu, u, CUFFT_FORWARD);
			cufftExecZ2Z(fftplan, fv, v, CUFFT_FORWARD);

			cublasZscal(handle, TGridSize, &inverseTGridSize, u, 1);
			cublasZscal(handle, TGridSize, &inverseTGridSize, v, 1);

			cublasZaxpy(handle, TGridSize, &onesixths, k[3], 1, u, 1);
			cublasZaxpy(handle, TGridSize, &onesixths, l[3], 1, v, 1);
			// vectorXXXKernel K2(4096,64) (......);
			vectorBoundResetKernel K2(4096, 64) (u);
			vectorBoundResetKernel K2(4096, 64) (v);



			//vectorThresholdKernel K2(4096, 64) (u);
			//vectorThresholdKernel K2(4096, 64) (v);
#if RECORD_HISTORY
			cudaMemcpy(uh[stepCount], u, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
#endif // RECORD_HISTORY

			

		}
		cudaMemcpy(ua[a], u, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaMemcpy(va[a], v, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time: %3.3f ms \n", elapsedTime);
	}
	cudaMemcpy(uu, u, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(tt, t, TGridSize * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hfu, fu, TGridSize * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	printf("GPU work done!\n");
	FILE* f;
	f = fopen("u.txt", "w");
	for (int i = 0; i < TGridSize; i++) {
		fprintf(f, "%e\n", abs(uu[i]));
	}
	fclose(f);

	f = fopen("t.txt", "w");
	for (int i = 0; i < TGridSize; i++) {
		fprintf(f, "%e\n", tt[i]);
	}
	fclose(f);

	f = fopen("fu.txt", "w");
	for (int i = 0; i < TGridSize; i++) {
		fprintf(f, "%e\n", abs(hfu[i]));
	}
	fclose(f);

	// Check r
	cudaMemcpy(tt, r, TGridSize * sizeof(double), cudaMemcpyDeviceToHost);
	f = fopen("r.txt", "w");
	for (int i = 0; i < TGridSize; i++) {
		fprintf(f, "%e\n", tt[i]);
	}
	fclose(f);

	cudaMemcpy(uu, fr, TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	f = fopen("fr.txt", "w");
	for (int i = 0; i < TGridSize; i++) {
		fprintf(f, "%e\n", (uu[i].y));
	}
	fclose(f);


	saveSimulationEnvironments(NULL);
	PrintVairableZ(fcr, 2 * TGridSize, "fcr.txt");
	PrintVairableZ(fcv, 2 * TGridSize, "fcv.txt");
	PrintVairableD(cv, 2*TGridSize, "cv.txt");
	PrintVairableD(cu, 2 * TGridSize, "cu.txt");
	PrintVairableD(cr, 2 * TGridSize, "cr.txt");
	PrintVairableD(tvd[0],  TGridSize, "tvd0.txt");
	PrintVairableZ(k[0], TGridSize, "k0.csv");
	PrintVairableZ(k[1], TGridSize, "k1.csv");
	PrintVairableZ(k[2], TGridSize, "k2.csv");
	PrintVairableZ(k[3], TGridSize, "k3.csv");
	


	auto fu = fopen("ua.zbin", "wb"); // u 
	auto fv = fopen("va.zbin", "wb"); // u 

	for (int ii = 0; ii < AngleCount; ii++)
	{
		cudaMemcpy(uu, ua[ii], TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		fwrite(uu, sizeof(cuDoubleComplex), TGridSize, fu);
		cudaMemcpy(uu, va[ii], TGridSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		fwrite(uu, sizeof(cuDoubleComplex), TGridSize, fv);
		//for (int jj = 0; jj < TGridSize; jj++)
		//{
		//	if (isnan(abs(uu[jj]))) uu[jj] = { -1,0 };
		//	
		//	fprintf(fx, "%e,", uu[jj].x);
		//	fprintf(fy, "%e,", uu[jj].y);
		//}
		//fprintf(fx, "\n");
		//fprintf(fy, "\n");
		//fflush(fx);
		//fflush(fy);
	}
	fclose(fu);
	fclose(fv);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.

	// Cleanup & shutdown
	cufftDestroy(fftplan);
	cufftDestroy(fftplan2);
	cufftDestroy(fftplan2I);
	cublasDestroy(handle);
	GPUMemoryClear();
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(uu);
	free(tt);
	free(hfu);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

void GPUMemoryAllocate()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&u, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&v, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&u2, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&u0, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&v2, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&u3, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&v3, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&t, TGridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&fu, TGridSize * sizeof(cufftDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&fv, TGridSize * sizeof(cufftDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&fr, TGridSize * sizeof(cufftDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&r, TGridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&cu, 2 * TGridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset(cu, 0, 2 * TGridSize * sizeof(double));

	cudaStatus = cudaMalloc((void**)&cr, 2 * TGridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset(cr, 0, 2 * TGridSize * sizeof(double));
	cudaStatus = cudaMalloc((void**)&cv, 2 * TGridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset(cv, 0, 2 * TGridSize * sizeof(double));

	cudaStatus = cudaMalloc((void**)&fcr, 2 * TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&fcu, 2 * TGridSize * sizeof(cufftDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&fcv, 2 * TGridSize * sizeof(cufftDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < tempVariableCount; i++)
	{
		cudaStatus = cudaMalloc((void**)&tv[i], TGridSize * sizeof(cufftDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&tvd[i], TGridSize * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		cudaStatus = cudaMalloc((void**)&k[i], TGridSize * sizeof(cufftDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&l[i], TGridSize * sizeof(cuDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}
	cudaStatus = cudaMalloc((void**)&dxvector, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dyvector, TGridSize * sizeof(cuDoubleComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
#if RECORD_HISTORY
	for (int i = 0; i < ZActualStep; i++)
	{
		cudaStatus = cudaMalloc((void**)&uh[i], TGridSize * sizeof(cufftDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}
#endif
	for (int i = 0; i < AngleCount; i++)
	{
		cudaStatus = cudaMalloc((void**)&ua[i], TGridSize * sizeof(cufftDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&va[i], TGridSize * sizeof(cufftDoubleComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}
	return;
Error:
	GPUMemoryClear();
	return;
}

void GPUMemoryClear()
{
	cudaFree(u);
	cudaFree(v);
	cudaFree(u2);
	cudaFree(u0);
	cudaFree(v2);
	cudaFree(u3);
	cudaFree(v3);
	cudaFree(t);
	cudaFree(fu);
	cudaFree(fv);
	cudaFree(fr);
	cudaFree(r);
	cudaFree(cu);
	cudaFree(cv);
	cudaFree(cr);
	cudaFree(fcu);
	cudaFree(fcv);
	cudaFree(fcr);
	for (int i = 0; i < tempVariableCount; i++)
	{
		cudaFree(tv[i]);
		cudaFree(tvd[i]);
	}
	for (int i = 0; i < 4; i++)
	{
		cudaFree(k[i]);
		cudaFree(l[i]);
	}
	cudaFree(dxvector);
	cudaFree(dyvector);
#if RECORD_HISTORY
	for (int i = 0; i < ZActualStep; i++)
	{
		cudaFree(uh[i]);
	}
#endif
	for (int i = 0; i < AngleCount; i++)
	{
		cudaFree(ua[i]);
		cudaFree(va[i]);
	}
	return;
}

void NConstruct(cuDoubleComplex *u, cuDoubleComplex *v, double *r, cuDoubleComplex* dstu, cuDoubleComplex* dstv)
{
#ifdef DISPERSION_ONLY
	cudaMemset(dstu, 0, TGridSize * sizeof(cuDoubleComplex));
	cudaMemset(dstv, 0, TGridSize * sizeof(cuDoubleComplex));
	return;
#endif
	
	// dstu&dstv
	vectorProductV1Kernel K2(4096, 128) (u,v, tv[2],tv[1],tv[0],tv[3],tvd[0],tvd[1]);
	// dstu
	cublasZaxpy(handle, TGridSize, &twothirds, tv[0], 1, tv[1], 1);
	cudaMemcpy(cu + TGridSize / 2, tvd[0], TGridSize * sizeof(double), cudaMemcpyDeviceToDevice);
	cufftExecD2Z(fftplan2, cu, fcu);

	// dstv
	cublasZaxpy(handle, TGridSize, &twothirds, tv[2], 1, tv[3], 1);
	cudaMemcpy(cv + TGridSize / 2, tvd[1], TGridSize * sizeof(double), cudaMemcpyDeviceToDevice);
	cufftExecD2Z(fftplan2, cv, fcv);
	// dstu&dstv
	vectorProductV3Kernel K2(4096, 64)(fcr, fcu, fcv);
	// dstu
	cufftExecZ2D(fftplan2I, fcu, cu);
	cublasDscal(handle, 2 * TGridSize, &inverseT2GridSize, cu, 1);
	cublasDscal(handle, 2 * TGridSize, &TStep, cu, 1);
	cudaMemcpy(tvd[0], cu + TGridSize, TGridSize * sizeof(double), cudaMemcpyDeviceToDevice);
#ifndef NO_RAMAN
	vectorProductAddKernel K2(4096, 64) (tv[1], tvd[0], u, tv[0]);
#else
	cublasZcopy(handle, TGridSize, tv[1], 1, tv[0], 1);
#endif
#ifndef NO_DIFFERENTIAL
	vectorDifferentialKernel K2(4096, 64) (tv[0], tv[1]);
	cublasZdscal(handle, TGridSize, &inverseTStep, tv[1], 1);
	cublasZaxpy(handle, TGridSize, &cromega0i, tv[1], 1, tv[0], 1);
#endif
	vectorLinearKernel K2(4096, 64) (tv[0], dstu, { 0 ,1 * gamma * ZStep }, { 0,0 });
	
	// dstv
	cufftExecZ2D(fftplan2I, fcv, cv);
	cublasDscal(handle, 2 * TGridSize, &inverseT2GridSize, cv, 1);
	cublasDscal(handle, 2 * TGridSize, &TStep, cv, 1);
	cudaMemcpy(tvd[1], cv + TGridSize, TGridSize * sizeof(double), cudaMemcpyDeviceToDevice);
#ifndef NO_RAMAN
	vectorProductAddKernel K2(4096, 64) (tv[3], tvd[1], v, tv[2]);
#else
	cublasZcopy(handle, TGridSize, tv[3], 1, tv[2], 1);
#endif
#ifndef NO_DIFFERENTIAL 
	vectorDifferentialKernel K2(4096, 64) (tv[2], tv[3]);
	cublasZdscal(handle, TGridSize, &inverseTStep, tv[3], 1);
	cublasZaxpy(handle, TGridSize, &cromega0i, tv[3], 1, tv[2], 1);
#endif
	vectorLinearKernel K2(4096, 64) (tv[2], dstv, { 0 ,1 * gamma * ZStep }, { 0,0 });
	
}

void PrintVairableD(double* gpumemory, int n, char* name)
{
	double * tt = (double*)malloc(n * sizeof(double));
	cudaMemcpy(tt, gpumemory, n * sizeof(double), cudaMemcpyDeviceToHost);
	FILE* f = fopen(name, "w");
	for (int i = 0; i < n; i++) {
		fprintf(f, "%e\n", tt[i]);
	}
	fclose(f);
	free(tt);
}

void PrintVairableZ(cuDoubleComplex* gpumemory, int n, char* name)
{
	cuDoubleComplex * tt = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
	cudaMemcpy(tt, gpumemory,  n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	FILE* f = fopen(name, "w");
	for (int i = 0; i < n; i++) {
		fprintf(f, "%e\n", abs(tt[i]));
	}
	fclose(f);
	free(tt);
}

void PrintVairableZA(cuDoubleComplex* gpumemory, int n, char* name)
{
	cuDoubleComplex * tt = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
	cudaMemcpy(tt, gpumemory, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	FILE* f = fopen(name, "w");
	for (int i = 0; i < n; i++) {
		fprintf(f, "%e,%e\n", tt[i].x,tt[i].y);
	}
	fclose(f);
	free(tt);
}

template <class T>
void SaveVariableBinary(T* v, int count, char* filename)
{
	auto f = fopen(filename, "wb");
	fwrite(v, sizeof(T), count, f);
	fclose(f);
}