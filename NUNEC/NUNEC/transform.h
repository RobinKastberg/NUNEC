#pragma once
struct mat4 {
	float c[16];
};
struct vec2 {
	float x, y;
};
struct vec3 {
	float x, y, z;
};
struct vec4 {
	float x, y, z, w;
};
float dot(struct vec3 * A, struct vec3 * B);
void mat4_identity(struct mat4 *self);
void mat4_debug(struct mat4 *self);
void vec3_normalize(struct vec3 *self);
#define vec3_neg(A) vec3 {-A.x, -A.y, -A.z}
struct vec3 vec3_cross(struct vec3 * __restrict A, struct vec3 * __restrict B);
void vec3_cross(struct vec3 * __restrict A, struct vec3 * __restrict B, struct vec3 * __restrict C);
void mat4_mul(struct mat4 * __restrict self, struct mat4 * __restrict other);
struct vec4 mat4_mul(struct mat4 * __restrict self, struct vec4 * __restrict other);
void mat4_mul(float * __restrict A, float * __restrict B, float * __restrict C);
void mat4_mul(struct mat4 * __restrict A, struct mat4 * __restrict B, struct mat4 * __restrict C);
void mat4_translate(struct mat4 * __restrict A, float x, float y, float z);
void mat4_lookat(struct mat4 * __restrict self, struct vec3 * __restrict pos, struct vec3 * __restrict lookat, struct vec3 * __restrict up);
struct vec3 vec3_new(float a, float b, float c);
void mat4_perspective(struct mat4 * __restrict self, float fov, float aspect, float nr, float fr);
void mat4_ortho(struct mat4 * __restrict self, float w, float h, float nr, float fr);
void mat4_inverse(const float * __restrict m, float * __restrict invOut);
void mat4_scale(struct mat4 * __restrict self, float scale);