#define _CRT_SECURE_NO_WARNINGS
#include "transform.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <math.h>
#define GLEW_STATIC
#include "glew.h"
#include <GL/gl.h>

#define WINDOWS


void swap();

#ifdef WINDOWS
__declspec(align(16)) struct cell {
#else
#include <GL/glx.h>
#include <strings.h>
#include <X11/keysym.h>
#include <X11/Xutil.h>
#include <X11/Xlib.h>
    struct cell {
#endif
        // H-field
        float Hx, Hy, Hz, __1;
        float CHx, CHy, CHz, __2;
        float CEx, CEy, CEz, __3;
        // D-field
        float Dx, Dy, Dz, __4;
        // E-field
        float Ex, Ey, Ez, __5;

        // Material parameters
        float muxx, muyy, muzz, mat;
        float epsxx, epsyy, epszz, pec;

        // precomputed constants
        float Mhx1, Mhx2, Mhx3, Mhx4;
        float Mhy1, Mhy2, Mhy3, Mhy4;
        float Mhz1, Mhz2, Mhz3, Mhz4;

        float Mdx1, Mdx2, Mdx3, Mdx4;
        float Mdy1, Mdy2, Mdy3, Mdy4;
        float Mdz1, Mdz2, Mdz3, Mdz4;

        // PML Integrals

        float ICex, ICey, ICez, __6;
        float IChx, IChy, IChz, __7;

        float Ihx, Ihy, Ihz, __8;
        float Idx, Idy, Idz, __9;
#ifdef LINUX
    } __attribute__((aligned(16)));
#else
};
#endif


struct mat4 M;
struct mat4 V;
int width = 640;
int height = 480;
struct mat4 MV;
struct mat4 P;
struct mat4 MVP;
float lookrot = 0.0;
float radius = 2.0;
float looktilt = 0.5;

const char *vert_shader = \
                          "#version 330 core\n"\
                          "layout(location = 0) in vec4 in_Vertex;"\
                          "varying out vec3 position;"\
                          "uniform mat4 MVP;"\
                          "uniform mat4 M;"\
                          "void main() {"\
                          "gl_Position = MVP * in_Vertex;"\
                          "position = vec3(in_Vertex.xyz);"\
                          "}";
const char *frag_shader = \
                          "#version 330 core\n"\
                          "varying in vec3 position;\n"\
                          "uniform vec3 cameraPosition;\n"\
                          "uniform sampler3D myTextureSampler;\n"\
                          "void main() {\n"\
                          "vec3 ray_dir = normalize(cameraPosition - position);\n"\
                          "vec3 ray_pos = position;\n"\
                          "vec3 pos111 = vec3(1.0, 1.0, 1.0);\n"\
                          "vec3 pos000 = vec3(0.0, 0.0, 0.0);\n"\
                          "const float sample_step = 0.01;\n"\
                          "const float val_threshold = 1;\n"\
                          "vec4 color;\n"\
                          "vec4 frag_color = vec4(0.0, 0.0, 0.0, 0.0);\n"\
                          "    do\n"\
                          "    {\n"\
                          "        // note: \n"\
                          "        // - ray_dir * sample_step can be precomputed\n"\
                          "        // - we assume the volume has a cube-like shape\n"\
                          "\n"\
                          "        ray_pos += ray_dir * sample_step;\n"\
                          "\n"\
                          "        // break out if ray reached the end of the cube.\n"\
                          "        if (max(max(ray_pos.x, ray_pos.y),ray_pos.z) > 1.0 || min(min(ray_pos.x, ray_pos.y),ray_pos.z) < -1.0)\n"\
                          "            break;\n" //  0.3125 + 0.375*
                          "		 if(length(ray_pos - cameraPosition) < 1)\n"
                          "			break;\n"
                          "        float E =  texture(myTextureSampler, (vec3(1)+ray_pos)/2.0).g;\n"\
                              "        float H =  texture(myTextureSampler,(vec3(1)+ray_pos)/2.0).b;\n"\
                              "        float mat =  texture(myTextureSampler, (vec3(1)+ray_pos)/2.0).a;\n"\
                              "		 float density = 10*sample_step*texture(myTextureSampler,(vec3(1)+ray_pos)/2.0).r;\n"\
                              "\n"\
                              "        color.rgb = clamp(E*vec3(1,0,0) + H*vec3(0,0,1) + mat*vec3(0,1,0),0,1);\n"\
                              "		 color.a = density;\n"\
                              "        frag_color.rgb = frag_color.rgb * (1.0 - color.a) + color.rgb * color.a;\n"\
                              ""
                              "    }\n"\
                                  "    while(true);\n"\
                                  "if (frag_color == vec4(0.0, 0.0, 0.0, 0.0))\n"\
                                  "discard;\n"\
                                  "else\n"\
                                  "gl_FragColor = vec4(frag_color.rgb, 1.0);\n"\
                                  "};\n";
static const GLfloat g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
    1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f
};

const int Nx = 60;
const int Ny = 100;
const int Nz = 16;
#define bsize ((Nx)*(Ny)*(Nz)*sizeof(struct cell))
int nsteps = 100;
int ks = 32;

float c0 = 1;
float frmax = 1e6;
float dx = 1e-10; //  c0 / (frmax * 100);
float dy = 1e-10; //  c0 / (frmax * 100);
float dz = 1e-10; //  c0 / (frmax * 100);
float dt;
float eps = 1;
float mu = 1;

float Ca, Cb, Da, Db;

struct cell (*sim)[Nx][Ny][Nz];
struct cell(*sim_gpu)[Nx][Ny][Nz];

int curstep = 0;

GLuint textureID;
GLuint VertexArrayID;
GLuint vertexbuffer;
GLenum my_program;
GLenum my_vertex_shader;
GLenum my_fragment_shader;
GLenum my_compute_shader;
GLenum my_compute_program;

float tex2D[Nx][Ny][Nz][4];
vec3 lookat = vec3{ 0, 0, 0 };
vec3 up = vec3{ 0, 0, 1 };
int L = 16;
float pml(int k)
{
    /*
    int NXLO = L - 1;
    int NXHI = ke - 1 - L;
    if (k < (L))
        k = (k - L);
    else if (k >(ke - 1 - L))
        k = k - (ke - 1 - L);
    else
        k = 0;

    k = abs(k);
    float s = (float)k/L;
    */
    return 0;
    //return (eps / (2*dt))*s*s*s;
}
GLuint ssbo;

void init()
{
	printf("INIT()\n");
    sim = (cell (*)[Nx][Ny][Nz])malloc(bsize);
    //omp_set_dynamic(0);
    //omp_set_num_threads(4);
    glGenBuffers(1, &ssbo);
	printf("glgenbuffers()\n");


    dt = dx / (100.0f * c0);
    //dt = 1;
    //dt = 1.6678e-10;
    //dt = dx;
    memset(sim, 0, bsize);

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int z = 0; z < Nz; z++) {
                float px = 0, py = 0, pz = 0;
                float rhoHx = 0, rhoHy = 0, rhoHz = 0;
                float rhoDx = 0, rhoDy = 0, rhoDz = 0;
                /*
                   if (x == 0 || x == ke - 1 || y == 0 || y == ke - 1 || z == 0 || z == ke - 1)
                   {
                   rhoHx = sim[x][y][z].pmlHx = 0;
                   rhoHy = sim[x][y][z].pmlHy = 0;
                   rhoHz = sim[x][y][z].pmlHz = 0;

                   rhoDx = sim[x][y][z].pmlDx = 0;
                   rhoDy = sim[x][y][z].pmlDy = 0;
                   rhoDz = sim[x][y][z].pmlDz = 0;
                   }*/
                //else {
                // TODO: Fungerar inte än.
                rhoHx = pml(x); //  pml(x);
                rhoHy = pml(y); //  pml(x);
                rhoHz = pml(z); //  pml(x);

                rhoDx = pml(x); //  pml(x);
                rhoDy = pml(y); //  pml(x);
                rhoDz = pml(z); //  pml(x);
                //}

                (*sim)[x][y][z].pec = 1;					
                (*sim)[x][y][z].epsxx = eps;
                (*sim)[x][y][z].epsyy = eps;
                (*sim)[x][y][z].epszz = eps;
                (*sim)[x][y][z].muxx = mu;
                (*sim)[x][y][z].muyy = mu;
                (*sim)[x][y][z].muzz = mu;
                //(*sim)[x][y][z].mat = (pml(y)+ pml(x) + pml(z))*0.5;
				/*
                if (z <= 2) {
                    (*sim)[x][y][z].epsxx = 2.2*eps;
                    (*sim)[x][y][z].epsyy = 2.2*eps;
                    (*sim)[x][y][z].epszz = 2.2*eps;
                    (*sim)[x][y][z].mat = 0.2;
                    (*sim)[x][y][z].pec = 1;

                }
                if(z == 3 && abs(x-30) < 16 && abs(y-50) < 20) {
                    (*sim)[x][y][z].pec = 0;
                    (*sim)[x][y][z].mat = 2;
                }
                if(z == 3 && (x >= 20 && x <= 25) && y < 50) {
                    (*sim)[x][y][z].pec = 0;
                    (*sim)[x][y][z].mat = 2;
                }
				*/





                double Mhx0 = (1.0 / dt) + ((rhoHy + rhoHz) / (2.0 * eps)) + rhoHy*rhoHz*dt / (4.0 * eps*eps);
                double Mhy0 = (1.0 / dt) + ((rhoHx + rhoHz) / (2.0 * eps)) + rhoHx*rhoHz*dt / (4.0 * eps*eps);
                double Mhz0 = (1.0 / dt) + ((rhoHx + rhoHy) / (2.0 * eps)) + rhoHx*rhoHy*dt / (4.0 * eps*eps);

                (*sim)[x][y][z].Mhx1 = (1.0 / Mhx0) * ((1.0 / dt) - ((rhoHy + rhoHz) / (2.0 * eps)) - rhoHy*rhoHz*dt / (4.0 * eps*eps));
                (*sim)[x][y][z].Mhy1 = (1.0 / Mhy0) * ((1.0 / dt) - ((rhoHx + rhoHz) / (2.0 * eps)) - rhoHx*rhoHz*dt / (4.0 * eps*eps));
                (*sim)[x][y][z].Mhz1 = (1.0 / Mhz0) * ((1.0 / dt) - ((rhoHx + rhoHy) / (2.0 * eps)) - rhoHx*rhoHy*dt / (4.0 * eps*eps));

                (*sim)[x][y][z].Mhx2 = -c0 / (Mhx0 * (*sim)[x][y][z].muxx);
                (*sim)[x][y][z].Mhy2 = -c0 / (Mhy0 * (*sim)[x][y][z].muyy);
                (*sim)[x][y][z].Mhz2 = -c0 / (Mhz0 * (*sim)[x][y][z].muzz);

                (*sim)[x][y][z].Mhx3 = - (c0*dt*rhoHx) / (Mhx0*eps*(*sim)[x][y][z].muxx);
                (*sim)[x][y][z].Mhy3 = - (c0*dt*rhoHy) / (Mhy0*eps*(*sim)[x][y][z].muyy);
                (*sim)[x][y][z].Mhz3 = - (c0*dt*rhoHz) / (Mhz0*eps*(*sim)[x][y][z].muzz);

                (*sim)[x][y][z].Mhx4 = -(dt*rhoHy*rhoHz) / (Mhx0*eps*eps);
                (*sim)[x][y][z].Mhy4 = -(dt*rhoHx*rhoHz) / (Mhy0*eps*eps);
                (*sim)[x][y][z].Mhz4 = -(dt*rhoHy*rhoHx) / (Mhz0*eps*eps);

                if (x == 0 && y == 0 && z == 0)
                {
                    printf("Mhx0: %e\n", (float)Mhx0);
                    printf("Mhx1: %e\n", (float)(*sim)[x][y][z].Mhx1);
                    printf("Mhx2: %e\n", (float)(*sim)[x][y][z].Mhx2);
                    printf("Mhx3: %e\n", (float)(*sim)[x][y][z].Mhx3);
                    printf("Mhx4: %e\n", (float)(*sim)[x][y][z].Mhx4);
                    printf("Mhx4(taljare): %e\n", (dt*rhoHy*rhoHz));
                    printf("Mhx4(namnare): %e\n", (Mhz0*eps*eps));


                    printf("dt: %e\n", dt);
                    printf("1/dt: %e\n", 1.0 / dt);
                    printf("rhoHx: %e\n", rhoHx);
                    printf("rhoHy: %e\n", rhoHx);
                    printf("rhoHz: %e\n", rhoHx);
                    printf("eps: %e\n", eps);
                    printf("c0: %e\n", c0);

                }




                float Mdx0 = (1.0 / dt) + ((rhoDy + rhoDz) / (2 * eps)) + rhoDy*rhoDz*dt / (4 * eps*eps);
                float Mdy0 = (1.0 / dt) + ((rhoDx + rhoDz) / (2 * eps)) + rhoDx*rhoDz*dt / (4 * eps*eps);
                float Mdz0 = (1.0 / dt) + ((rhoDx + rhoDy) / (2 * eps)) + rhoDx*rhoDy*dt / (4 * eps*eps);

                (*sim)[x][y][z].Mdx1 = (1.0 / Mdx0) * ((1.0 / dt) - ((rhoDy + rhoDz) / (2 * eps)) - rhoDy*rhoDz*dt / (4 * eps*eps));
                (*sim)[x][y][z].Mdy1 = (1.0 / Mdy0) * ((1.0 / dt) - ((rhoDx + rhoDz) / (2 * eps)) - rhoDx*rhoDz*dt / (4 * eps*eps));
                (*sim)[x][y][z].Mdz1 = (1.0 / Mdz0) * ((1.0 / dt) - ((rhoDx + rhoDy) / (2 * eps)) - rhoDx*rhoDy*dt / (4 * eps*eps));

                (*sim)[x][y][z].Mdx2 =  c0 / (Mdx0);
                (*sim)[x][y][z].Mdy2 =  c0 / (Mdy0);
                (*sim)[x][y][z].Mdz2 =  c0 / (Mdz0);

                (*sim)[x][y][z].Mdx3 =  (c0*dt*rhoDx) / (Mdx0*eps);
                (*sim)[x][y][z].Mdy3 =  (c0*dt*rhoDy) / (Mdy0*eps);
                (*sim)[x][y][z].Mdz3 =  (c0*dt*rhoDz) / (Mdz0*eps);

                (*sim)[x][y][z].Mdx4 = -(dt*rhoDy*rhoDz) / (Mdx0*eps*eps);
                (*sim)[x][y][z].Mdy4 = -(dt*rhoDx*rhoDz) / (Mdy0*eps*eps);
                (*sim)[x][y][z].Mdz4 = -(dt*rhoDy*rhoDx) / (Mdz0*eps*eps);

                if (x == 0 && y == 0 && z == 0)
                {
                    printf("Mdx0: %e\n", (float)Mdx0);
                    printf("Mdx1: %e\n", (float)(*sim)[x][y][z].Mdx1);
                    printf("Mdx2: %e\n", (float)(*sim)[x][y][z].Mdx2);
                    printf("Mdx3: %e\n", (float)(*sim)[x][y][z].Mdx3);
                    printf("Mdx4: %e\n", (float)(*sim)[x][y][z].Mdx4);
                    printf("Mdx4(taljare): %e\n", (dt*rhoDy*rhoDz));
                    printf("Mdx4(namnare): %e\n", (Mdz0*eps*eps));

                    printf("rhoDx: %e\n", rhoDx);
                    printf("rhoDy: %e\n", rhoDx);
                    printf("rhoDz: %e\n", rhoDx);

                }
            }
        }
    }


    mat4_identity(&M);

    mat4_perspective(&P, 90.0, (float)width / height, 1, 100);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_3D, textureID);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, Nx, Ny, Nz, 0, GL_RGBA, GL_FLOAT, tex2D);
    

    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    glGenBuffers(1, &vertexbuffer);
    // The following commands will talk about our 'vertexbuffer' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Give our vertices to OpenGL.
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);



    // Create Shader And Program Objects
    my_program = glCreateProgram();
    my_compute_program = glCreateProgram();
    my_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    my_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    my_compute_shader = glCreateShader(GL_COMPUTE_SHADER);

    char shader_buf[1 << 16];
    char *ptr = shader_buf;
    FILE *fp = fopen("compute.comp", "r");
    int read = fread(shader_buf, 1, 1 << 16, fp);
	shader_buf[read] = '\0';
    // Load Shader Sources
    glShaderSource(my_vertex_shader, 1, &vert_shader, NULL);
    glShaderSource(my_fragment_shader, 1, &frag_shader, NULL);
    glShaderSource(my_compute_shader, 1, &ptr, NULL);

    // Compile The Shaders
    glCompileShader(my_vertex_shader);
    GLint isCompiled = 0;
    char errorLog[512];
    int maxLength = 512;
    glGetShaderiv(my_vertex_shader, GL_COMPILE_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(my_vertex_shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        glGetShaderInfoLog(my_vertex_shader, maxLength, &maxLength, &errorLog[0]);
        printf("%s\n", errorLog);

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        //glDeleteShader(shader); // Don't leak the shader.
        return;
    }
    glCompileShader(my_fragment_shader);
    glGetShaderiv(my_fragment_shader, GL_COMPILE_STATUS, &isCompiled);

    if (isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(my_fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        glGetShaderInfoLog(my_fragment_shader, maxLength, &maxLength, &errorLog[0]);
        printf("%s\n", errorLog);

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        //glDeleteShader(shader); // Don't leak the shader.
        return;
    }

    glCompileShader(my_compute_shader);
    glGetShaderiv(my_compute_shader, GL_COMPILE_STATUS, &isCompiled);

    if (isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(my_compute_shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        glGetShaderInfoLog(my_compute_shader, maxLength, &maxLength, &errorLog[0]);
        printf("%s\n", errorLog);

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        //glDeleteShader(shader); // Don't leak the shader.
        return;
    }

	glGetShaderiv(my_compute_shader, GL_LINK_STATUS, &isCompiled);

	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(my_compute_shader, GL_INFO_LOG_LENGTH, &maxLength);

		// The maxLength includes the NULL character
		glGetShaderInfoLog(my_compute_shader, maxLength, &maxLength, &errorLog[0]);
		printf("%s\n", errorLog);

		// Provide the infolog in whatever manor you deem best.
		// Exit with failure.
		//glDeleteShader(shader); // Don't leak the shader.
		return;
	}

    // Attach The Shader Objects To The Program Object
    glAttachObjectARB(my_program, my_vertex_shader);
    glAttachObjectARB(my_program, my_fragment_shader);

    glAttachObjectARB(my_compute_program, my_compute_shader);

    // Link The Program Object
    glLinkProgramARB(my_program);
    glLinkProgramARB(my_compute_program);

    // Use The Program Object Instead Of Fixed Function OpenGL

    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT);
    //glEnable(GL_BLEND);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, bsize, sim,  GL_MAP_READ_BIT);
    //glBufferData(GL_SHADER_STORAGE_BUFFER, bsize, sim, GL_STATIC_DRAW);
    free(sim);
}

FILE *pipe;
void peek(int x, int y, int z)
{
    struct cell peek;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (Nz*Ny*x + Nz*y + z)*sizeof(struct cell), sizeof(struct cell), (void *)&peek);
    printf("CELL: (%02d,%02d,%02d)\n", x, y, z);
    printf("CHx: %f\n", peek.CHx);
    printf("CHy: %e\n", peek.CHy);
    printf("CHz: %e\n", peek.CHz);
    printf("CEx: %f\n", peek.CEx);
    printf("CEy: %e\n", peek.CEy);
    printf("CEz: %e\n", peek.CEz);
    printf("Ex: %f\n", peek.Ex);
    printf("Ey: %e\n", peek.Ey);
    printf("Ez: %e\n", peek.Ez);
    printf("Dx: %e\n", peek.Dx);
    printf("Dy: %e\n", peek.Dy);
    printf("Dz: %e\n", peek.Dz);
    printf("Hx: %e\n", peek.Hx);
    printf("Hy: %e\n", peek.Hy);
    printf("Hz: %e\n", peek.Hz);
    printf("Mdx1: %e\n", peek.Mdx1);
    printf("Mdx2: %e\n", peek.Mdx2);
    printf("Mdx3: %e\n", peek.Mdx3);
    printf("Mdx4: %e\n", peek.Mdx4);
    printf("Mhx1: %e\n", peek.Mhx1);
    printf("Mhx2: %e\n", peek.Mhx2);
    printf("Mhx3: %e\n", peek.Mhx3);
    printf("Mhx4: %e\n", peek.Mhx4);
    printf("ICex: %e\n", peek.ICex);
    printf("ICey: %e\n", peek.ICey);
    printf("ICez: %e\n", peek.ICez);
    printf("IChx: %e\n", peek.IChx);
    printf("IChy: %e\n", peek.IChy);
    printf("IChz: %e\n", peek.IChz);
    printf("Ihx: %e\n", peek.Ihx);
    printf("Ihy: %e\n", peek.Ihy);
    printf("Ihz: %e\n", peek.Ihz);
    printf("Idx: %e\n", peek.Ihx);
    printf("Idy: %e\n", peek.Ihy);
    printf("Idz: %e\n", peek.Ihz);
    printf("muxx: %e\n", peek.muxx);
    printf("muyy: %e\n", peek.muyy);
    printf("muzz: %e\n", peek.muzz);
    printf("epsxx: %e\n", peek.epsxx);
    printf("epsyy: %e\n", peek.epsyy);
    printf("epszz: %e\n", peek.epszz);
}
void swap();

void draw()
{
    float t = dt*curstep; 
    printf("t: %e\n", t);
    //glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    //peek(30,50,8);
    glUseProgram(my_compute_program);
    {
        glUniform1i(glGetUniformLocation(my_compute_program, "sim"), 0);
        glUniform1i(glGetUniformLocation(my_compute_program, "ks"), ks);
        glUniform1i(glGetUniformLocation(my_compute_program, "Nx"), Nx);
        glUniform1i(glGetUniformLocation(my_compute_program, "Ny"), Ny);
        glUniform1i(glGetUniformLocation(my_compute_program, "Nz"), Nz);
        glUniform1i(glGetUniformLocation(my_compute_program, "curstep"), curstep);

        glUniform1f(glGetUniformLocation(my_compute_program, "dx"), dx);
        glUniform1f(glGetUniformLocation(my_compute_program, "dy"), dy);
        glUniform1f(glGetUniformLocation(my_compute_program, "dz"), dz);
        glUniform1f(glGetUniformLocation(my_compute_program, "dt"), dt);
        glUniform1f(glGetUniformLocation(my_compute_program, "c0"), c0);

        glBindImageTexture(0, textureID, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA8);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
        glDispatchCompute(Nx, Ny, Nz/16);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
    glUseProgram(my_program);
    {
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_3D, textureID);
        // Give the image to OpenGL
        /*glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, ke, ke, ke, GL_RGBA, GL_FLOAT, tex2D);*/
        /*glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);*/
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //glGenerateMipmap(GL_TEXTURE_3D);

        glEnable(GL_TEXTURE_3D);
        vec3 pos = vec3{ radius * cosf(lookrot), radius * sinf(lookrot), looktilt };
        mat4_lookat(&V, &pos, &lookat, &up);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void*)0            // array buffer offset
                );

        mat4_mul(&M, &V, &MV);
        mat4_mul(&MV, &P, &MVP);

        glUniformMatrix4fv(glGetUniformLocation(my_program, "MVP"), 1, GL_FALSE, &MVP.c[0]);
        glUniformMatrix4fv(glGetUniformLocation(my_program, "M"), 1, GL_FALSE, &M.c[0]);

        glUniform3fv(glGetUniformLocation(my_program, "cameraPosition"), 1, &pos.x);

        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 12 * 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        glDisableVertexAttribArray(0);
        //step();
        //step();
        //MessageBoxA(0, (char*)glGetString(GL_VERSION), "OPENGL VERSION", 0);

        swap();

        curstep++;
        printf("step: %05d\n", curstep);
#ifndef WINDOWS
        if(curstep == 1)
        {
            pipe = popen("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 60 -i - -an -vf vflip output.mp4", "w");
        } else if(curstep > 1) {
            char buf[3*640*480];
            glReadPixels(0,0,width, height, GL_RGB, GL_UNSIGNED_BYTE, buf);
            fwrite(buf, 3, width*height, pipe);
        }
#endif
    }
}
void resize()
{
	glViewport(0, 0, width, height);
	mat4_perspective(&P, 90.0, (float)width / height, 0.001, 100);
}
#ifndef WINDOWS
Display *dpy;
Window glwin;
void swap()
{
        glXSwapBuffers(dpy, glwin);
}

static Bool WaitForMapNotify(Display *d, XEvent *e, char *arg) 
{ 
    if ((e->type == MapNotify) && (e->xmap.window == (Window)arg)) { 
        return GL_TRUE; 
    } 
    return GL_FALSE; 
} 

int main(int argc, char **argv)
{
    XVisualInfo    *vi; 
    Colormap        cmap; 
    XSetWindowAttributes swa; 
    GLXContext      cx; 
    XEvent          event; 
    GLboolean       needRedraw = GL_FALSE, recalcModelView = GL_TRUE; 
    int             dummy; 
	int attributes[] = {
		GLX_RGBA,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_DEPTH_SIZE, 16,
		0
	};

    dpy = XOpenDisplay(NULL); 
    if (dpy == NULL){ 
        fprintf(stderr, "could not open display\n"); 
        exit(1); 
    } 

    if(!glXQueryExtension(dpy, &dummy, &dummy)){ 
        fprintf(stderr, "could not open display"); 
        exit(1); 
    } 

    /* find an OpenGL-capable Color Index visual with depth buffer */ 
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributes); 
    if (vi == NULL) { 
        fprintf(stderr, "could not get visual\n"); 
        exit(1); 
    } 

    /* create an OpenGL rendering context */ 
    cx = glXCreateContext(dpy, vi,  None, GL_TRUE); 
    if (cx == NULL) { 
        fprintf(stderr, "could not create rendering context\n"); 
        exit(1); 
    } 

    /* create an X colormap since probably not using default visual */ 
    cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),  
            vi->visual, AllocNone); 
    swa.colormap = cmap; 
    swa.border_pixel = 0; 
    swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask; 
    glwin = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, width, 
            height, 0, vi->depth, InputOutput, vi->visual, 
            CWBorderPixel | CWColormap | CWEventMask, &swa); 
    XSetStandardProperties(dpy, glwin, "xogl", "xogl", None, argv,  
            argc, NULL); 

	Atom wm_delete = XInternAtom( dpy, "WM_DELETE_WINDOW", 1 );
	XSetWMProtocols( dpy, glwin, &wm_delete, 1 );
    glXMakeCurrent(dpy, glwin, cx); 
    XMapWindow(dpy, glwin); 
    XIfEvent(dpy,  &event,  WaitForMapNotify,  (char *)glwin); 

    glewExperimental = true;
    int ginit = glewInit();
    printf("glewInit: %d\n", ginit);
    printf("%s\n", glewGetErrorString(ginit));

    init();
    resize(); 

    /* Animation loop */ 
    while (1) { 
        KeySym key; 

        while (XPending(dpy)) { 
            XNextEvent(dpy, &event); 
            switch (event.type) { 
                case KeyPress: 
                    XLookupString((XKeyEvent *)&event, NULL, 0, &key, NULL); 
                    switch (key) { 
                        case XK_Left: 
                            lookrot += 0.1f;
                            break; 
                        case XK_Right: 
                            lookrot -= 0.1f;
                            break; 
                        case XK_Up: 
                            looktilt += 0.1f;
                            break; 
                        case XK_Down: 
                            looktilt -= 0.1f;
                            break; 
                        case XK_plus: 
                            radius -= 0.1f;
                            break; 
                        case XK_minus: 
                            radius += 0.1f;
                            break; 
                    } 
                    break; 
                case ConfigureNotify: 
                    width = event.xconfigure.width;
                    height = event.xconfigure.height;
                    resize(); 
                    break; 
				case ClientMessage:
					if ( !strcmp( XGetAtomName( dpy, event.xclient.message_type ), "WM_PROTOCOLS" ) )
						return 0;
					break;
				default:
					break;
            } 
        } 
		draw();
    }
    return 0;
}
#endif