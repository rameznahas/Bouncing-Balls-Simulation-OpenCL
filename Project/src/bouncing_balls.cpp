#include <time.h>
#include <glew.h>
#include <freeglut.h>
#include <cl.h>
#include <cl_gl.h>
#include <string>
#include <random>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#define MAX_INFO_LENGTH 1024
#define DEBUG_LOG_BUFFER_SIZE 16384
#define WWIDTH 800
#define WHEIGHT 800
#define UPDATE_FREQ 1.f / 30
#define BALL_COUNT 10
#define MIN_RADIUS 0.05f
//#define PI 3.141592f
//#define DEGREE_TO_RAD PI / 180
#define NUM_POINTS 360
#define WORK_GROUP_SIZE 256

struct ball {
	ball() = default;
	ball(float r, cl_float2 c, cl_float2 vel, int m)
		:
		radius(r),
		mass(m)
	{
		if (radius == 0.05f) {
			color[0] = 0.5f;
			color[1] = 1.f;
			color[2] = 0.5f;
		}
		else if (radius == 0.1f) {
			color[0] = 0.5f;
			color[1] = 0.5f;
			color[2] = 1.f;
		}
		else {
			color[0] = 1.f;
			color[1] = 0.5f;
			color[2] = 0.5f;
		}
		center[0] = c.x;
		center[1] = c.y;
		velocity[0] = vel.x;
		velocity[1] = vel.y;
	}

	float color[3];
	float center[2];
	float velocity[2];
	float radius;
	int mass;
};

GLuint vbo;

ball* balls = nullptr;
unsigned int* pairs = nullptr;
size_t balls_count, pairs_count;
size_t balls_size, pairs_size;

cl_context context = nullptr;
cl_device_id device = nullptr;
cl_command_queue cmd_q = nullptr;
cl_program program = nullptr;
cl_mem d_balls = nullptr, d_pairs = nullptr, d_vbo = nullptr;
cl_kernel wall_bounce = nullptr, ball_bounce = nullptr, update_vbo = nullptr;
cl_int status = CL_SUCCESS;

clock_t previous_t = 0, current_t = 0;
float delta_t = UPDATE_FREQ;

// forward declaration
void update();

void create_context() {
	status = CL_SUCCESS;
	cl_uint num_platforms;
	cl_platform_id* platforms;

	status = clGetPlatformIDs(0, nullptr, &num_platforms);
	if (status != CL_SUCCESS || num_platforms < 1) {
		std::cout << "Couldn't find any OpenCL platforms." << std::endl;
		return;
	}
	std::cout << "Found " << num_platforms << " platform(s)." << std::endl << std::endl;

	platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, nullptr);

	char info[MAX_INFO_LENGTH];

	for (unsigned int i = 0; i < num_platforms; ++i) {
		std::cout << "Platform (" << (i + 1) << ")" << std::endl;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(info), info, nullptr);
		std::cout << "  Vendor:\t" << info << std::endl;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(info), info, nullptr);
		std::cout << "  Name:\t\t" << info << std::endl;

		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(info), info, nullptr);
		std::cout << "  Version:\t" << info << std::endl << std::endl;
	}

	std::cout << "Platform choice: ";
	int platform_num;
	std::cin >> platform_num;
	std::cout << std::endl;

	cl_platform_id platform = platforms[platform_num - 1];
	delete[] platforms;

	cl_uint num_devices;
	cl_device_id* devices;

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
	if (status != CL_SUCCESS || num_devices < 1) {
		std::cout << "Couldn't find any devices." << std::endl;
		return;
	}
	std::cout << "Found " << num_devices << " device(s)." << std::endl << std::endl;

	devices = new cl_device_id[num_devices];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr);

	for (unsigned int i = 0; i < num_devices; ++i) {
		std::cout << "Device (" << (i + 1) << ")" << std::endl;
		
		cl_device_type device_type;
		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);

		if (device_type & CL_DEVICE_TYPE_CPU)
			std::cout << "  Type:\t\t\tCL_DEVICE_TYPE_CPU" << std::endl;
		else if (device_type & CL_DEVICE_TYPE_GPU)
			std::cout << "  Type:\t\t\tCL_DEVICE_TYPE_GPU" << std::endl;
		else if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
			std::cout << "  Type:\t\t\tCL_DEVICE_TYPE_ACCELERATOR" << std::endl;
		else 
			std::cout << "  Type:\t\t\tCL_DEVICE_TYPE_DEFAULT" << std::endl;

		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(info), info, nullptr);
		std::cout << "  Name:\t\t\t" << info << std::endl;

		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(info), info, nullptr);
		std::cout << "  Vendor:\t\t" << info << std::endl;

		cl_uint max_compute_units;
		clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, nullptr);
		std::cout << "  Max Compute Unit:\t" << max_compute_units << std::endl << std::endl;
	}

	std::cout << "Device choice: ";
	int device_num;
	std::cin >> device_num;
	std::cout << std::endl;

	cl_device_id device = devices[device_num - 1];
	delete[] devices;

	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0
	};

	context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
}

cl_int create_clgl_buffers() {
	status = CL_SUCCESS;

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, balls_count * NUM_POINTS * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	d_vbo = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vbo, &status);
	if (status != CL_SUCCESS || d_vbo == nullptr) {
		std::cout << "Failed to associate CL buffer to GL buffer." << std::endl;
		return status;
	}

	d_balls = clCreateBuffer(context, CL_MEM_READ_WRITE, balls_size, nullptr, &status);
	if (status != CL_SUCCESS || d_balls == nullptr) {
		std::cout << "Failed to allocate a buffer on device." << std::endl;
		return status;
	}
	status = clEnqueueWriteBuffer(cmd_q, d_balls, CL_TRUE, 0, balls_size, balls, 0, nullptr, nullptr);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to write data to device memory." << std::endl;
		return status;
	}

	d_pairs = clCreateBuffer(context, CL_MEM_READ_WRITE, pairs_size, nullptr, &status);
	if (status != CL_SUCCESS || d_pairs == nullptr) {
		std::cout << "Failed to allocate a buffer on device." << std::endl;
		return status;
	}

	status = clEnqueueWriteBuffer(cmd_q, d_pairs, CL_TRUE, 0, pairs_size, pairs, 0, nullptr, nullptr);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to write data to device memory." << std::endl;
		return status;
	}

	return status;
}

void create_program(cl_uint num_devices, const char* file_name) {
	std::ifstream kernel(file_name, std::ifstream::in);
	if (!kernel.is_open()) {
		std::cout << "Failed to open kernel file." << std::endl;
		return;
	}

	std::ostringstream buff;
	buff << kernel.rdbuf();

	std::string s = buff.str();
	const char* kernels = s.c_str();

	program = clCreateProgramWithSource(context, 1, (const char**)&kernels, nullptr, &status);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to create CL program from source file " << file_name << std::endl;
		return;
	}

	status = clBuildProgram(program, num_devices, &device, nullptr, nullptr, nullptr);
	if (status != CL_SUCCESS) {
		char log[DEBUG_LOG_BUFFER_SIZE];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);

		std::cout << "Failed to build CL program." << std::endl;
		std::cerr << log;
		clReleaseProgram(program);
	}
}

cl_int create_kernels() {
	status = CL_SUCCESS;

	wall_bounce = clCreateKernel(program, "wall_bounce", &status);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to create kernel from program." << std::endl;
		return status;
	}

	status = clSetKernelArg(wall_bounce, 0, sizeof(cl_mem), &d_balls);
	status |= clSetKernelArg(wall_bounce, 1, sizeof(float), &delta_t);
	status |= clSetKernelArg(wall_bounce, 2, sizeof(unsigned int), &balls_count);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to set kernel args." << std::endl;
		return status;
	}

	ball_bounce = clCreateKernel(program, "ball_bounce", &status);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to create kernel from program." << std::endl;
		return status;
	}

	status = clSetKernelArg(ball_bounce, 0, sizeof(cl_mem), &d_pairs);
	status |= clSetKernelArg(ball_bounce, 1, sizeof(cl_mem), &d_balls);
	status |= clSetKernelArg(ball_bounce, 2, sizeof(unsigned int), &pairs_count);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to set kernel args." << std::endl;
		return status;
	}

	update_vbo = clCreateKernel(program, "update_vbo", &status);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to create kernel from program." << std::endl;
		return status;
	}

	status = clSetKernelArg(update_vbo, 0, sizeof(cl_mem), &d_balls);
	status |= clSetKernelArg(update_vbo, 1, sizeof(cl_mem), &d_vbo);
	status |= clSetKernelArg(update_vbo, 2, sizeof(unsigned int), &balls_count);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to set kernel args." << std::endl;
		return status;
	}

	return status;
}

void init(int argc, char** argv) {
	//////////////////////////init display//////////////////////////
	glutInit(&argc, argv);
	glutInitWindowPosition(-1, -1);
	glutInitWindowSize(WWIDTH, WHEIGHT);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
	glutCreateWindow("Bouncing Balls Simulation");
	glewInit();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glutDisplayFunc(update);
	glutIdleFunc(update);
	////////////////////////////////////////////////////////////////

	///////////////////////////init balls///////////////////////////
	balls_count = argc > 1 ? std::stoi(argv[1]) : BALL_COUNT;
	balls = new ball[balls_count];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> rad(1, 3);
	std::uniform_real_distribution<float> vel(-1.f, 1.f);

	for (unsigned int i = 0; i < balls_count; ++i) {
		int test = rad(gen);
		float radius = MIN_RADIUS * test; // random radius

		float ur_bound = radius - 1;
		float ll_bound = 1 - radius;

		std::uniform_real_distribution<float> coord(ur_bound, ll_bound); // so we dont get balls out of bounds
		cl_float2 center = { coord(gen), coord(gen) };

		int weight = (int)(radius * 100.0f);
		cl_float2 velocity = { vel(gen), vel(gen) };
		balls[i] = ball(radius, center, velocity, weight);
	}

	balls_size = balls_count * sizeof(ball);
	////////////////////////////////////////////////////////////////

	///////////////////////////init pairs///////////////////////////
	for (unsigned int i = 0; i < balls_count; ++i) {
		for (unsigned int j = (i + 1); j < balls_count; ++j) {
			++pairs_count;
		}
	}

	pairs = new unsigned int[2 * pairs_count];

	int count = 0;
	for (unsigned int i = 0; i < balls_count; ++i) {
		for (unsigned int j = (i + 1); j < balls_count; ++j) {
			pairs[count++] = i;
			pairs[count++] = j;
		}
	}

	pairs_size = pairs_count * sizeof(unsigned int) * 2;
	////////////////////////////////////////////////////////////////
}

void draw() {
	glClearColor(0.25f, 0.25f, 0.25f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	for (unsigned int i = 0; i < balls_count; ++i) {
		ball& ball = balls[i];
		glColor4f(ball.color[0], ball.color[1], ball.color[2], 0.25f);
		glDrawArrays(GL_POLYGON, i * NUM_POINTS * 2, NUM_POINTS * 2);
	}
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();
}

void update() {
	//update current clock time
	current_t = clock();
	delta_t = (float)(current_t - previous_t) / CLOCKS_PER_SEC;

	// don't draw if delta_t is faster than 30 fps
	if (delta_t < UPDATE_FREQ) return;

	// store last draw time
	previous_t = current_t;

	//glFinish();
	clEnqueueNDRangeKernel(cmd_q, wall_bounce, 1, nullptr, &balls_count, &balls_count, 0, nullptr, nullptr);
	clEnqueueNDRangeKernel(cmd_q, ball_bounce, 1, nullptr, &pairs_count, &pairs_count, 0, nullptr, nullptr);
	clEnqueueAcquireGLObjects(cmd_q, 1, &d_vbo, 0, nullptr, nullptr);
	clEnqueueNDRangeKernel(cmd_q, update_vbo, 1, nullptr, &balls_count, &balls_count, 0, nullptr, nullptr);
	clEnqueueReleaseGLObjects(cmd_q, 1, &d_balls, 0, nullptr, nullptr);
	//clFinish(cmd_q);

	draw();
}

void cleanup() {
	if (balls) delete[] balls;
	if (pairs) delete[] pairs;
	if (d_balls) clReleaseMemObject(d_balls);
	if (d_pairs) clReleaseMemObject(d_pairs);
	if (cmd_q) clReleaseCommandQueue(cmd_q);
	if (wall_bounce) clReleaseKernel(wall_bounce);
	if (ball_bounce) clReleaseKernel(ball_bounce);
	if (program) clReleaseProgram(program);
	if (context) clReleaseContext(context);
	// add opengl cleanup
}

int main(int argc, char** argv) {
	init(argc, argv);

	create_context();
	if (!context) {
		std::cout << "Failed to create an OpenCL context." << std::endl;
		cleanup();
		std::exit(1);
	}

	status = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to get device from context." << std::endl;
		cleanup();
		std::exit(1);
	}

	cmd_q = clCreateCommandQueue(context, device, 0, &status);
	if (status != CL_SUCCESS) {
		std::cout << "Failed to get device from context." << std::endl;
		cleanup();
		std::exit(1);
	}

	status = create_clgl_buffers();
	if (status != CL_SUCCESS) {
		cleanup();
		std::exit(1);
	}

	create_program(1, "./src/bouncing_balls.cl");
	if (!program) {
		cleanup();
		std::exit(1);
	}

	status = create_kernels();
	if (status != CL_SUCCESS) {
		cleanup();
		std::exit(1);
	}
	
	glutMainLoop();

	cleanup();

	return 0;
}
