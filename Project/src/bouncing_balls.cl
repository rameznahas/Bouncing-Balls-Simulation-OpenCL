#define NUM_POINTS 360
#define PI 3.141592f

constant float DEGREE_TO_RAD = PI / 180;
constant int NUM_FLOATS = NUM_POINTS * 2;

struct ball {
	float color[3];
	float center[2];
	float velocity[2];
	float radius;
	int mass;
};

/*
	Handles the ball-wall computation.
*/
__kernel void wall_bounce(__global struct ball* d_balls, float delta_t, unsigned int balls_count) {
	 int id = get_global_id(0);
		if (id < balls_count) {
			__global struct ball* current = &d_balls[id];

			current->velocity[1] += delta_t * -1.5f;
			current->center[0] += delta_t * current->velocity[0];
			current->center[1] += delta_t * current->velocity[1];

			float t_wall = 1.f - current->radius;
			float b_wall = current->radius - 1.f;
			float r_wall = t_wall;
			float l_wall = b_wall;

			if (current->center[0] > r_wall) {
				current->center[0] = r_wall;
				current->velocity[0] *= -1.f;
			}
			else if (current->center[0] < l_wall) {
				current->center[0] = l_wall;
				current->velocity[0] *= -1.f;
			}

			if (current->center[1] > t_wall) {
				current->center[1] = t_wall;
				current->velocity[1] *= -1.f;
			}
			else if (current->center[1] < b_wall) {
				current->center[1] = b_wall;
				current->velocity[1] *= -1.f;
			}
		}
}

/*
	Handles the ball-ball computation.
*/
__kernel void ball_bounce(__global unsigned int* d_pairs, __global struct ball* d_balls, unsigned int pairs_count) {
	unsigned int id = get_global_id(0);
	if (id < pairs_count) {
		unsigned int stride = 2 * id;
		__global struct ball* current = &d_balls[d_pairs[stride]];
		__global struct ball* other = &d_balls[d_pairs[stride + 1]];

		float min_dist = current->radius + other->radius;

		// check for aabb overlap
		// if true, balls are close enough, computation is worth it.
		if (current->center[0] + min_dist > other->center[0]
			&& current->center[1] + min_dist > other->center[1]
			&& other->center[0] + min_dist > current->center[0]
			&& other->center[1] + min_dist > current->center[1]) {

			float c_x = current->center[0] - other->center[0];
			float c_y = current->center[1] - other->center[1];
			float c = pow(c_x, 2.f) + pow(c_y, 2.f);

			// balls are close enough, but it does not mean they have collided.
			// check for ball collision.
			// if true, collision occured, handle it
			if (c <= pow(min_dist, 2.f)) {
				float dist = sqrt(c);
				float overlap = 0.5f * (dist - current->radius - other->radius);

				float dir_x = c_x / dist;
				float dir_y = c_y / dist;

				current->center[0] -= overlap * dir_x;
				current->center[1] -= overlap * dir_y;
				other->center[0] += overlap * dir_x;
				other->center[1] += overlap * dir_y;

				float v_x = current->velocity[0] - other->velocity[0];
				float v_y = current->velocity[1] - other->velocity[1];
				int m = current->mass + other->mass;
				float mag = pow(dist, 2.f);
				float dot_vc = v_x * c_x + v_y * c_y;
				float ratio = 2.f * dot_vc / (m * mag);

				current->velocity[0] -= (other->mass * ratio * c_x);
				current->velocity[1] -= (other->mass * ratio * c_y);
				other->velocity[0] += (current->mass * ratio * c_x);
				other->velocity[1] += (current->mass * ratio * c_y);
			}
		}
	}
}

/*
	Updates the vbo to be used by OpenGL to draw the new values computed earlier.
*/
__kernel void update_vbo(__global struct ball* d_balls, __global float* d_vbo, unsigned int balls_count) { 
	int id = get_global_id(0);
	if (id < balls_count) { 
		__global struct ball* current = &d_balls[id];

		int idx = id * NUM_FLOATS;

		for (int j = 0; j < NUM_POINTS; ++j) {
			float angle = j * DEGREE_TO_RAD;
			d_vbo[idx++] = current->radius * cos(angle) + current->center[0]; // x-coord
			d_vbo[idx++] = current->radius * sin(angle) + current->center[1]; // y-coord
		}
	}
}