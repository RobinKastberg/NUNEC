#include "transform.h"
#include "math.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
void mat4_identity(struct mat4 *self)
{
	self->c[0] = 1.0f; self->c[1] = 0.0f; self->c[2] = 0.0f; self->c[3] = 0.0f;
	self->c[4] = 0.0f; self->c[5] = 1.0f; self->c[6] = 0.0f; self->c[7] = 0.0f;
	self->c[8] = 0.0f; self->c[9] = 0.0f; self->c[10] = 1.0f; self->c[11] = 0.0f;
	self->c[12] = 0.0f; self->c[13] = 0.0f; self->c[14] = 0.0f; self->c[15] = 1.0f;
}

void mat4_mul(float * __restrict A, float * __restrict B, float * __restrict C) {
	for (int x = 0; x < 4; x++) { // row number of output
		for (int y = 0; y < 4; y++) { // column number of output
			C[4 * x + y] = 0;
			for (int z = 0; z < 4; z++) { // four elements are added for this output
				C[4 * x + y] += A[4 * x + z] * B[4 * z + y];
			}
		}
	}
}
void mat4_mul(struct mat4 * __restrict A, struct mat4 * __restrict B, struct mat4 * __restrict C)
{
	mat4_mul(A->c, B->c, C->c);
}
void mat4_mul(struct mat4 * __restrict self, struct mat4 * __restrict other)
{
	struct mat4 result;
	mat4_mul(self->c, other->c, result.c);
	memcpy(self, result.c, 16 * sizeof(float));
}
struct vec4 mat4_mul(mat4 * __restrict self, vec4 * __restrict other)
{
	struct vec4 ret;
	ret.x  = other->x * self->c[0] + other->y * self->c[4] + other->z * self->c[8] + other->w * self->c[12];
	ret.y  = other->x * self->c[1] + other->y * self->c[5] + other->z * self->c[9] + other->w * self->c[13];
	ret.z  = other->x * self->c[2] + other->y * self->c[6] + other->z * self->c[10] + other->w * self->c[14];
	ret.w  = other->x * self->c[3] + other->y * self->c[7] + other->z * self->c[11] + other->w * self->c[15];
	return ret;

}
void mat4_debug(struct mat4 *self)
{
	char out[100];
	for (int i = 0; i < 4; i++)
	{
		snprintf(out, 100, "%f %f %f %f\n", self->c[4 * i + 0], self->c[4 * i + 1], self->c[4 * i + 2], self->c[4 * i + 3]);
	}
}

float dot(struct vec3 * A, struct vec3 * B)
{
	return A->x * B->x + A->y * B->y + A->z* B->z;
}

void vec3_normalize(vec3 * self)
{
	float nrm = sqrtf(dot(self, self));
	self->x /= nrm;
	self->y /= nrm;
	self->z /= nrm;
}
void vec3_cross(struct vec3 * __restrict A, struct vec3 * __restrict B, struct vec3 * __restrict C)
{
	C->x = A->y * B->z - A->z * B->y;
	C->y = A->z * B->x - A->x * B->z;
	C->z = A->x * B->y - A->y * B->x;
}

struct vec3 vec3_cross(struct vec3 * __restrict A, struct vec3 * __restrict B)
{
	struct vec3 C;
	vec3_cross(A, B, &C);
	return C;
}
struct vec3 vec3_new(float a, float b, float c)
{
	struct vec3 ret;
	ret.x = a;
	ret.y = b;
	ret.z = c;

	return ret;
}
void mat4_ortho(struct mat4 * __restrict self, float w, float h, float nr, float fr)
{
	self->c[0] = 1 / w; self->c[1] = 0; self->c[2] = 0; self->c[3] = 0.0f;
	self->c[4] = 0; self->c[5] = 1/ h; self->c[6] = 0; self->c[7] = 0.0f;
	self->c[8] = 0; self->c[9] = 0; self->c[10] = 2 / (nr - fr); self->c[11] = 0.0f;
	self->c[12] = 0; self->c[13] = 0; self->c[14] = (fr + nr) / (nr - fr); self->c[15] = 1.0f;
}
void mat4_perspective(struct mat4 * __restrict self, float fov, float aspect, float nr, float fr)
{
	float top = nr*tanf(3.141592*fov / 360.0);
	float bottom = -top;
	float right = top*aspect;
	float left = -right;

	self->c[0] = 2*nr/(right-left); self->c[1] = 0; self->c[2] = 0; self->c[3] = 0.0f;
	self->c[4] = 0; self->c[5] = 2*nr/(top-bottom); self->c[6] = 0; self->c[7] = 0.0f;
	self->c[8] = 0; self->c[9] = 0; self->c[10] = (nr + fr) / (nr - fr); self->c[11] = -1;
	self->c[12] = 0; self->c[13] = 0; self->c[14] = 2 * fr*nr / (nr - fr); self->c[15] = 0.0f;
}

void mat4_scale(struct mat4 * __restrict self, float scale)
{
	self->c[0] = scale; self->c[1] = 0; self->c[2] = 0; self->c[3] = 0.0f;
	self->c[4] = 0; self->c[5] = scale; self->c[6] = 0; self->c[7] = 0.0f;
	self->c[8] = 0; self->c[9] = 0; self->c[10] = scale; self->c[11] = 0;
	self->c[12] = 0; self->c[13] = 0; self->c[14] = 0; self->c[15] = 1.0f;
}
void mat4_lookat(struct mat4 * __restrict self, struct vec3 * __restrict pos, struct vec3 * __restrict lookat, struct vec3 * __restrict up)
{
	struct vec3 zaxis = { pos->x - lookat->x,
						  pos->y - lookat->y,
					      pos->z - lookat->z };
	vec3_normalize(&zaxis);

	struct vec3 xaxis;
	vec3_cross(up, &zaxis, &xaxis);
	vec3_normalize(&xaxis);

	struct vec3 yaxis;
	vec3_cross(&zaxis, &xaxis, &yaxis);

	self->c[0] = xaxis.x; self->c[1] = yaxis.x; self->c[2] = zaxis.x; self->c[3] = 0.0f;
	self->c[4] = xaxis.y; self->c[5] = yaxis.y; self->c[6] = zaxis.y; self->c[7] = 0.0f;
	self->c[8] = xaxis.z; self->c[9] = yaxis.z; self->c[10] = zaxis.z; self->c[11] = 0.0f;
	self->c[12] = -dot(&xaxis, pos); self->c[13] = -dot(&yaxis, pos); self->c[14] = -dot(&zaxis, pos); self->c[15] = 1.0f;
}

void mat4_inverse(const float * __restrict m, float * __restrict invOut)
{
	double inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	assert(det != 0);
		

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;
}
void mat4_translate(struct mat4 * __restrict self, float x, float y, float z)
{
	self->c[0] = 1.0f; self->c[1] = 0; self->c[2] = 0; self->c[3] = 0;
	self->c[4] = 0; self->c[5] = 1.0f; self->c[6] = 0; self->c[7] = 0;
	self->c[8] = 0; self->c[9] = 0; self->c[10] = 1.0f; self->c[11] = 0;
	self->c[12] = x; self->c[13] = y; self->c[14] = z; self->c[15] = 1.0f;
}
