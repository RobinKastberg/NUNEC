#include "transform.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <math.h>
#define GLEW_STATIC
#include "glew.h"
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/keysym.h>
#include <X11/Xutil.h>
#include <X11/Xlib.h>



void swap();

#ifdef WINDOWS
__declspec(align(16)) struct cell {
#else
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
const char *compute_shader = \
                             "#version 430 \n\
                             struct cell {\n\
                                 vec3 H;\n\
                                     vec3 CH;\n\
                                     vec3 CE;\n\
                                     vec3 D;\n\
                                     vec3 E;\n\
                                     vec4 mu;\n\
                                     vec4 eps;\n\
                                     vec4 Mhx;\n\
                                     vec4 Mhy;\n\
                                     vec4 Mhz;\n\
                                     vec4 Mdx;\n\
                                     vec4 Mdy;\n\
                                     vec4 Mdz;\n\
                                     vec3 ICe;\n\
                                     vec3 ICh;\n\
                                     vec3 Ih;\n\
                                     vec3 Id;\n\
                             } ;\n"\
                             "layout(std140, binding = 2) coherent buffer shader_data {\n\
                             cell vertex[];\n\
                             };\n"\
                             "layout(local_size_x=1, local_size_y=1, local_size_z=16) in;\n"\
                             "layout(rgba8, location = 0) uniform image3D sim;\n"\
                             "uniform int ks;\n"\
                             "uniform int ke;\n"\
                             "uniform int curstep;\n"\
                             "uniform float dx;\n"\
                             "uniform float c0;\n"\
                             "uniform float dt;\n"\
                             "void main() {\n"\
                             "uint x = gl_GlobalInvocationID.x;\n"\
                             "uint y = gl_GlobalInvocationID.y;\n"\
                             "uint z = gl_GlobalInvocationID.z;\n"\
                             "uint off = ke * ke * x + ke * y + z;\n"\
                             "float plusXEy = (x == (ke - 1)) ? 0 : vertex[off + ke*ke].E.y;\n"\
                             "float plusXEz = (x == (ke - 1)) ? 0 : vertex[off + ke*ke].E.z;\n"\
                             "float plusYEz = (y == (ke - 1)) ? 0 : vertex[off + ke].E.z;\n"\
                             "float plusYEx = (y == (ke - 1)) ? 0 : vertex[off + ke].E.x;\n"\
                             "float plusZEy = (z == (ke - 1)) ? 0 : vertex[off + 1].E.y;\n"\
                             "float plusZEx = (z == (ke - 1)) ? 0 : vertex[off + 1].E.x;\n"\
                             "vertex[off].CE.x = -(plusZEy - vertex[off].E.y - plusYEz + vertex[off].E.z) / dx;\n"\
                             "vertex[off].CE.y = -(plusXEz - vertex[off].E.z - plusZEx + vertex[off].E.x) / dx;\n"\
                             "vertex[off].CE.z = -(plusYEx - vertex[off].E.x - plusXEy + vertex[off].E.y) / dx;\n"\
                             "memoryBarrierBuffer();\n"\
                             "barrier();\n"\
                             "//float source = 6*exp(-0.5*(curstep-250.0)*(curstep-250.0)/(100000));\n"
                             "//float Hsource = sin(2*3.1415*dt*curstep*1e7)*min(100/curstep,1);\n"
                             "//float Esource = sin(2*3.1415*dt*(curstep+0.5)*1e7)*min(100/curstep,1);\n"
                             "float sig = 1e-6;\n"
                             "float t = dt*curstep - 4e-6; \n"
                             "float t2 = dt*(curstep + 0.5) + dx/(2*c0) - 4e-6; \n"
                             "float Esource = 0.0001*(2.0/(sqrt(3*sig)*1.3313))*(1-((t/sig)*(t/sig)))*exp(-t*t/(2*sig*sig));\n"
                             "float Hsource = 0.0001*(2.0/(sqrt(3*sig)*1.3313))*(1-((t2/sig)*(t2/sig)))*exp(-t2*t2/(2*sig*sig));\n"
                             "float Ex_src = Esource;\n" // E-Source
                             "float Ey_src = Esource;\n" // H-Source
                             "float Hx_src = -Hsource;\n" // E-Source
                             "float Hy_src = Hsource;\n" // H-Source
                                 "if(z == 18 && x > 20 && y > 20 && x < 44 && y < 44) {"\
                                 "vertex[off].CE.x += Ey_src / dx;\n"\
                                 "vertex[off].CE.y -= Ex_src / dx;\n"\
                                 "}"

                             "barrier();\n"\
                                 "memoryBarrierBuffer();\n"\
                                 "barrier();\n"\
                                 "vertex[off].ICe += vertex[off].CE;\n"\
                                 "vertex[off].Ih += vertex[off].H; \n"\
                                 "vertex[off].H.x = dot(vertex[off].Mhx, vec4(vertex[off].H.x, \n\
                                 vertex[off].CE.x, \n\
                                 vertex[off].ICe.x, \n\
                                 vertex[off].Ih.x));\n"\
                                 "vertex[off].H.y = dot(vertex[off].Mhy, vec4(vertex[off].H.y, \n\
                                 vertex[off].CE.y, \n\
                                 vertex[off].ICe.y, \n\
                                 vertex[off].Ih.y));\n"\
                                 "vertex[off].H.z = dot(vertex[off].Mhz, vec4(vertex[off].H.z, \n\
                                 vertex[off].CE.z, \n\
                                 vertex[off].ICe.z, \n\
                                 vertex[off].Ih.z));\n"\
                                 "memoryBarrierBuffer();\n"\
                                 "float minXHy = (x == 0) ? 0 : vertex[off - ke*ke].H.y;\n"\
                                 "float minXHz = (x == 0) ? 0 : vertex[off - ke*ke].H.z;\n"\
                                 "float minYHz = (y == 0) ? 0 : vertex[off - ke].H.z;\n"\
                                 "float minYHx = (y == 0) ? 0 : vertex[off - ke].H.x;\n"\
                                 "float minZHy = (z == 0) ? 0 : vertex[off - 1].H.y;\n"\
                                 "float minZHx = (z == 0) ? 0 : vertex[off - 1].H.x;\n"\
                                 "vertex[off].CH.x = (minZHy - vertex[off].H.y - minYHz + vertex[off].H.z) / dx;\n"\
                                 "vertex[off].CH.y = (minXHz - vertex[off].H.z - minZHx + vertex[off].H.x) / dx;\n"\
                                 "vertex[off].CH.z = (minYHx - vertex[off].H.x - minXHy + vertex[off].H.y) / dx;\n"\
                                 "memoryBarrierBuffer();\n"\

                                 "if(z == 19 && x > 20 && y > 20 && x < 44 && y < 44) {"\
                                 "vertex[off].CH.x += Hy_src / dx;\n"\
                                 "vertex[off].CH.y -= Hx_src / dx;\n"\
                                 "}"

                                     "memoryBarrierBuffer();\n"\

                                     "vertex[off].ICh += vertex[off].CH;\n"\
                                         "vertex[off].Id += vertex[off].D; \n"\
                                         "vertex[off].D.x = dot(vertex[off].Mdx, vec4(vertex[off].D.x, \n\
                                         vertex[off].CH.x, \n\
                                         vertex[off].ICh.x, \n\
                                         vertex[off].Id.x));\n"\
                                         "vertex[off].D.y = dot(vertex[off].Mdy, vec4(vertex[off].D.y, \n\
                                         vertex[off].CH.y, \n\
                                         vertex[off].ICh.y, \n\
                                         vertex[off].Id.y));\n"\
                                         "vertex[off].D.z = dot(vertex[off].Mdz, vec4(vertex[off].D.z, \n\
                                         vertex[off].CH.z, \n\
                                         vertex[off].ICh.z, \n\
                                         vertex[off].Id.z));\n"\
                                         "float pec = vertex[off].eps.w;\n"\
                                         "memoryBarrierBuffer();\n"\
                                         "vertex[off].E = pec * vertex[off].D / vec3(vertex[off].eps);\n"\
                                         "vec3 E = vertex[off].E;\n"\
                                         "vec3 H = vertex[off].H;\n"\
                                         "vec3 P = 300*cross(E, H);\n"\
                                         "imageStore(sim, ivec3(gl_GlobalInvocationID), vec4(100*vertex[off].mu.w+log(1+length(P)), 20*log(1+length(E)), 20*log(1+length(H)), vertex[off].mu.w));\n"\
                                         "//imageStore(sim, ivec3(gl_GlobalInvocationID), vec4(100*vertex[off].mu.w+length(P), length(E), 0, 0));\n"\
                                         ""\
                                         "}";

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

#define ke 64
#define bsize ((ke)*(ke)*(ke)*sizeof(struct cell))
int nsteps = 100;
int ks = 32;

float c0 = 1;
float frmax = 1e6;
float dx = 1e-7; //  c0 / (frmax * 100);
float dt;
float eps = 1;
float mu = 1;

float Ca, Cb, Da, Db;

struct cell (*sim)[ke][ke][ke];
struct cell(*sim_gpu)[ke][ke][ke];

int curstep = 0;

GLuint textureID;
GLuint VertexArrayID;
GLuint vertexbuffer;
GLenum my_program;
GLenum my_vertex_shader;
GLenum my_fragment_shader;
GLenum my_compute_shader;
GLenum my_compute_program;

float tex2D[ke][ke][ke][4];
vec3 lookat = vec3{ 0, 0, 0 };
vec3 up = vec3{ 0, 0, 1 };
int L = 16;
float pml(int k)
{
    int NXLO = L - 1;
    int NXHI = ke - 1 - L;
    if (k < (L))
        k = (k - L);
    else if (k >(ke - 1 - L))
        k = k - (ke - 1 - L);
    else
        k = 0;

    k = abs(k);

    return pow(k / L, 3.0);
}
float pmlD(int k)
{
    int NXLO = L - 1 ;
    int NXHI = ke - 1 - L;
    if (k < (L))
        k = (k - L);
    else if (k > (ke - 1 - L))
        k = k - (ke - 1 - L);
    else
        k = 0;

    k = abs(k);

    return (eps / (2*dt))*pow(k/L, 3.0);
}
float pmlH(int k)
{
    int NXLO = L - 1;
    int NXHI = ke - 1 - L;
    if (k < (L))
        k = (k - L);
    else if (k >(ke - 1 - L))
        k = k - (ke - 1 - L);
    else
        k = 0;

    k = abs(k);

    return (eps / (2 *dt))*pow(k / L, 3.0);
}
GLuint ssbo;

void init()
{
	printf("INIT()\n");
    sim = (cell (*)[ke][ke][ke])malloc(bsize);
    //omp_set_dynamic(0);
    //omp_set_num_threads(4);
    glGenBuffers(1, &ssbo);
	printf("glgenbuffers()\n");


    dt = dx / (10*sqrtf(3.0f) * c0);
    //dt = 1.6678e-10;
    //dt = dx;
    bzero(sim, bsize);

    for (int x = 0; x < ke; x++) {
        for (int y = 0; y < ke; y++) {
            for (int z = 0; z < ke; z++) {
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
                rhoHx = pmlH(x); //  pml(x);
                rhoHy = pmlH(y); //  pml(x);
                rhoHz = pmlH(z); //  pml(x);

                rhoDx = pmlD(x); //  pml(x);
                rhoDy = pmlD(y); //  pml(x);
                rhoDz = pmlD(z); //  pml(x);
                //}

                (*sim)[x][y][z].pec = 1;					
                (*sim)[x][y][z].epsxx = 1;
                (*sim)[x][y][z].epsyy = 1;
                (*sim)[x][y][z].epszz = 1;
                (*sim)[x][y][z].muxx = 1;
                (*sim)[x][y][z].muyy = 1;
                (*sim)[x][y][z].muzz = 1;
                //(*sim)[x][y][z].mat = (pml(y)+ pml(x) + pml(z))*0.5;

                if ((x-ks)*(x-ks)/3.0 + 6*(y-ks)*(y-ks)+6*(z-ks)*(z-ks) < 100.0) {
                    (*sim)[x][y][z].epsxx = 1e5;
                    (*sim)[x][y][z].epsyy = 1e5;
                    (*sim)[x][y][z].epszz = 1e5;
                    (*sim)[x][y][z].mat = 1;
                    (*sim)[x][y][z].pec = 0;

                }






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
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, ke, ke, ke, 0, GL_RGBA, GL_FLOAT, tex2D);

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

    // Load Shader Sources
    glShaderSource(my_vertex_shader, 1, &vert_shader, NULL);
    glShaderSource(my_fragment_shader, 1, &frag_shader, NULL);
    glShaderSource(my_compute_shader, 1, &compute_shader, NULL);

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

void step()
{
    /*
#pragma omp parallel for
for (int x = 0; x < ke ; x++) {
for (int y = 0; y < ke - 1 ; y++) {
for (int z = 0; z < ke - 1 ; z++) {
sim[x][y][z].CEx = -(sim[x][y][z + 1].Ey - sim[x][y][z].Ey - sim[x][y + 1][z].Ez + sim[x][y][z].Ez) / dx;
}
sim[x][y][ke-1].CEx = -(0 - sim[x][y][ke-1].Ey - sim[x][y + 1][ke-1].Ez + sim[x][y][ke-1].Ez) / dx;
}
for (int z = 0; z < ke - 1; z++) {
sim[x][ke-1][z].CEx = -(sim[x][ke-1][z + 1].Ey - sim[x][ke-1][z].Ey - 0 + sim[x][ke-1][z].Ez) / dx;
}

sim[x][ke - 1][ke - 1].CEx = -(0 - sim[x][ke-1][ke-1].Ey - 0 + sim[x][ke-1][ke-1].Ez) / dx;

}
#pragma omp parallel for
for (int y = 0; y < ke; y++) {
for (int x = 0; x < ke - 1; x++) {
for (int z = 0; z < ke - 1; z++) {
sim[x][y][z].CEy = -(sim[x + 1][y][z].Ez - sim[x][y][z].Ez - sim[x][y][z + 1].Ex + sim[x][y][z].Ex) / dx;
}
sim[x][y][ke - 1].CEy = -(sim[x + 1][y][ke-1].Ez - sim[x][y][ke-1].Ez - 0 + sim[x][y][ke-1].Ex) / dx;
}
for (int z = 0; z < ke - 1; z++) {
sim[ke-1][y][z].CEy = -(0 - sim[ke-1][y][z].Ez - sim[ke-1][y][z + 1].Ex + sim[ke-1][y][z].Ex) / dx;
}

sim[ke - 1][y][ke - 1].CEy = -(0 - sim[ke-1][y][ke-1].Ez - 0 + sim[ke-1][y][ke-1].Ex) / dx;

}
#pragma omp parallel for
for (int z = 0; z < ke; z++) {
for (int x = 0; x < ke - 1; x++) {
for (int y = 0; y < ke - 1; y++) {
sim[x][y][z].CEz = -(sim[x][y + 1][z].Ex - sim[x][y][z].Ex - sim[x + 1][y][z].Ey + sim[x][y][z].Ey) / dx;
}
sim[x][ke-1][z].CEz = -(0 - sim[x][ke-1][z].Ex - sim[x + 1][ke-1][z].Ey + sim[x][ke-1][z].Ey) / dx;
}
for (int y = 0; y < ke - 1; y++) {
sim[ke - 1][y][z].CEz = -(sim[ke - 1][y + 1][z].Ex - sim[ke -1 ][y][z].Ex - 0 + sim[ke-1][y][z].Ey) / dx;
}

sim[ke-1][ke - 1][z].CEz = -(0 - sim[ke-1][ke-1][z].Ex - 0 + sim[ke-1][ke-1][z].Ey) / dx;

}
#pragma omp parallel for
for (int x = 0; x < ke; x++) {
for (int y = 0; y < ke; y++) {
for (int z = 0; z < ke; z++) {

sim[x][y][z].ICex += sim[x][y][z].CEx;
sim[x][y][z].ICey += sim[x][y][z].CEy;
sim[x][y][z].ICez += sim[x][y][z].CEz;


sim[x][y][z].Ihx += sim[x][y][z].Hx;
sim[x][y][z].Ihy += sim[x][y][z].Hy;
sim[x][y][z].Ihz += sim[x][y][z].Hz;
}
}
}
// CURL H
#pragma omp parallel for
for (int x = 0; x < ke; x++) {
for (int y = 0; y < ke; y++) {
for (int z = 0; z < ke; z++) {


sim[x][y][z].Hx = sim[x][y][z].Mhx1 * sim[x][y][z].Hx
+ sim[x][y][z].Mhx2 * sim[x][y][z].CEx
    + sim[x][y][z].Mhx3 * sim[x][y][z].ICex
        + sim[x][y][z].Mhx4 * sim[x][y][z].Ihx;

    sim[x][y][z].Hy = sim[x][y][z].Mhy1 * sim[x][y][z].Hy
        + sim[x][y][z].Mhy2 * sim[x][y][z].CEy
        + sim[x][y][z].Mhy3 * sim[x][y][z].ICey
        + sim[x][y][z].Mhy4 * sim[x][y][z].Ihy;

    sim[x][y][z].Hz = sim[x][y][z].Mhz1 * sim[x][y][z].Hz
        + sim[x][y][z].Mhz2 * sim[x][y][z].CEz
        + sim[x][y][z].Mhz3 * sim[x][y][z].ICez
        + sim[x][y][z].Mhz4 * sim[x][y][z].Ihz;


}
}
}


#pragma omp parallel for
for (int x = 0; x < ke; x++) {
    sim[x][0][0].CHx = (0 - sim[x][0][0].Hy - 0 + sim[x][0][0].Hz) / dx;

    for (int z = 1; z < ke; z++) {
        sim[x][0][z].CHx = (sim[x][0][z - 1].Hy - sim[x][0][z].Hy - 0 + sim[x][0][z].Hz) / dx;
    }
    for (int y = 1; y < ke; y++) {
        sim[x][y][0].CHx = (0 - sim[x][y][0].Hy - sim[x][y - 1][0].Hz + sim[x][y][0].Hz) / dx;
        for (int z = 1; z < ke; z++) {
            sim[x][y][z].CHx = (sim[x][y][z - 1].Hy - sim[x][y][z].Hy - sim[x][y - 1][z].Hz + sim[x][y][z].Hz) / dx;
        }
    }
}
#pragma omp parallel for
for (int y = 0; y < ke; y++) {
    sim[0][y][0].CHy = (0 - sim[0][y][0].Hz - 0 + sim[0][y][0].Hx) / dx;
    for (int z = 1; z < ke; z++) {
        sim[0][y][z].CHy = (0 - sim[0][y][z].Hz - sim[0][y][z - 1].Hx + sim[0][y][z].Hx) / dx;
    }
    for (int x = 1; x < ke; x++) {
        sim[x][y][0].CHy = (sim[x - 1][y][0].Hz - sim[x][y][0].Hz - 0 + sim[x][y][0].Hx) / dx;

        for (int z = 1; z < ke; z++) {
            sim[x][y][z].CHy = (sim[x - 1][y][z].Hz - sim[x][y][z].Hz - sim[x][y][z - 1].Hx + sim[x][y][z].Hx) / dx;
        }
    }
}
#pragma omp parallel for
for (int z = 0; z < ke; z++) {
    sim[0][0][z].CHz = (0 - sim[0][0][z].Hx - 0 + sim[0][0][z].Hy) / dx;

    for (int y = 1; y < ke; y++) {
        sim[0][y][z].CHz = (sim[0][y - 1][z].Hx - sim[0][y][z].Hx - 0 + sim[0][y][z].Hy) / dx;
    }
    for (int x = 1; x < ke; x++) {
        sim[x][0][z].CHz = (0 - sim[x][0][z].Hx - sim[x - 1][0][z].Hy + sim[x][0][z].Hy) / dx;

        for (int y = 1; y < ke; y++) {
            sim[x][y][z].CHz = (sim[x][y - 1][z].Hx - sim[x][y][z].Hx - sim[x - 1][y][z].Hy + sim[x][y][z].Hy) / dx;
        }
    }


}

#pragma omp parallel for
for (int x = 1; x < ke - 1; x++) {
    for (int y = 1; y < ke - 1; y++) {
        for (int z = 1; z < ke - 1; z++) {

            sim[x][y][z].IChx += sim[x][y][z].CHx;
            sim[x][y][z].IChy += sim[x][y][z].CHy;
            sim[x][y][z].IChz += sim[x][y][z].CHz;

            sim[x][y][z].Idx += sim[x][y][z].Dx;
            sim[x][y][z].Idy += sim[x][y][z].Dy;
            sim[x][y][z].Idz += sim[x][y][z].Dz;
        }
    }
}

int wl = 20;
/*
   if (curstep == 1)
   {
   jz[ks][ks][ks] = 10;
   jz[ks+1][ks][ks] = 15;
   }	else  if(curstep == 2){
   jz[ks][ks][ks] = -10;
   jz[ks + 1][ks][ks] = -15;
   }*/
/*
#pragma omp parallel for
for (int x = 0; x < ke ; x++) {
for (int y = 0; y < ke ; y++) {
for (int z = 0; z < ke ; z++) {

sim[x][y][z].Dx = sim[x][y][z].Mdx1 * sim[x][y][z].Dx
+ sim[x][y][z].Mdx2 * sim[x][y][z].CHx
+ sim[x][y][z].Mdx3 * sim[x][y][z].IChx
+ sim[x][y][z].Mdx4 * sim[x][y][z].Idx;


sim[x][y][z].Dy = sim[x][y][z].Mdy1 * sim[x][y][z].Dy
+ sim[x][y][z].Mdy2 * sim[x][y][z].CHy
+ sim[x][y][z].Mdy3 * sim[x][y][z].IChy
+ sim[x][y][z].Mdy4 * sim[x][y][z].Idy;

sim[x][y][z].Dz = sim[x][y][z].Mdz1 * sim[x][y][z].Dz
+ sim[x][y][z].Mdz2 * sim[x][y][z].CHz
+ sim[x][y][z].Mdz3 * sim[x][y][z].IChz
+ sim[x][y][z].Mdz4 * sim[x][y][z].Idz
- jz[x][y][z];

sim[x][y][z].Ex = sim[x][y][z].Dx / sim[x][y][z].epsxx;
sim[x][y][z].Ey = sim[x][y][z].Dy / sim[x][y][z].epsyy;
sim[x][y][z].Ez = sim[x][y][z].Dz / sim[x][y][z].epszz;



}
}
}

int x = ks;
float minimume= 99999;
float maximume = -99999;
float minimumh = 99999;
float maximumh = -99999;

for (int x = 0 ; x < ke; x++)
{
for (int y = 0; y < ke; y++) {
for (int z = 0; z < ke; z++) {
if (y == ks && z == ks)
{
tex2D[x - 1][y - 1][z - 1][0] = 0;
tex2D[x - 1][y - 1][z - 1][1] = 0;
continue;
}
tex2D[x][y][z][0] = 1 + sim[x][y][z].Mhx2;

//float emag = log(1+sqrtf(ex[x][y][z] * ex[x][y][z] + ey[x][y][z] * ey[x][y][z] + ez[x][y][z] * ez[x][y][z]));
//float hmag = log(1+sqrtf(hx[x][y][z] * hx[x][y][z] + hy[x][y][z] * hy[x][y][z] + hz[x][y][z] * hz[x][y][z]));
vec3 E = vec3{ sim[x][y][z].Ex, sim[x][y][z].Ey, sim[x][y][z].Ez };
vec3 H = vec3{ sim[x][y][z].Hx, sim[x][y][z].Hy, sim[x][y][z].Hx };
vec3 c = vec3_cross(&E, &H);
float mag = sqrtf(dot(&c, &c));
tex2D[x][y][z][0] = 100 * log(1+mag);

tex2D[x][y][z][1] = 100 * log(1 + sqrtf(dot(&E, &E)));
tex2D[x][y][z][2] = 100 * log(1 + sqrtf(dot(&H, &H)));				
}
}
}

 */
}
FILE *pipe;
void peek(int x, int y, int z)
{
    struct cell peek;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (ke*ke*x + ke*y + z)*sizeof(struct cell), sizeof(struct cell), (void *)&peek);
    printf("CELL: (%02d,%02d,%02d)\n", x, y, z);
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
}

void draw()
{
    float t = dt*curstep - 1e-7;
    float sig = 1e-8;
    float Esource = (2.0/(sqrt(3.0*sig)*1.3313))*(1.0-((t/sig)*(t/sig)))*exp(-t*t/(2.0*sig*sig));
    printf("t: %e\n", t);
    printf("ES: %e\n", Esource);
    //glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    //peek(ks-1, ks-1, 20);
    glUseProgram(my_compute_program);
    {
        glUniform1i(glGetUniformLocation(my_compute_program, "sim"), 0);
        glUniform1i(glGetUniformLocation(my_compute_program, "ks"), ks);
        glUniform1i(glGetUniformLocation(my_compute_program, "ke"), ke);
        glUniform1i(glGetUniformLocation(my_compute_program, "curstep"), curstep);

        glUniform1f(glGetUniformLocation(my_compute_program, "dx"), dx);
        glUniform1f(glGetUniformLocation(my_compute_program, "dt"), dt);
        glUniform1f(glGetUniformLocation(my_compute_program, "c0"), c0);

        glBindImageTexture(0, textureID, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA8);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
        glDispatchCompute(64, 64, 4);
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
        if(curstep == 1)
        {
            pipe = popen("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 60 -i - -an -vf vflip output.mp4", "w");
        } else if(curstep > 1) {
            char buf[3*640*480];
            glReadPixels(0,0,width, height, GL_RGB, GL_UNSIGNED_BYTE, buf);
            fwrite(buf, 3, width*height, pipe);
        }
    }
}
Display *dpy;
Window glwin;
void swap()
{
        glXSwapBuffers(dpy, glwin);
}
void resize()
{
    glViewport(0, 0, width, height);
    mat4_perspective(&P, 90.0, (float)width / height, 0.001, 100);
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
