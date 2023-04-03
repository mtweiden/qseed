OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
u3(pi/2,0,pi) q[0];
u3(pi/2,-pi,-pi) q[1];
u3(pi/2,-pi,-pi) q[2];
u3(pi/2,0,pi) q[3];
u3(pi/2,-pi,-pi) q[4];
u3(pi/2,-pi,-pi) q[5];
u3(pi/2,0,pi) q[6];
u3(pi/2,-pi,-pi) q[7];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
u3(pi/2,-pi,-pi) q[10];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
u3(pi/2,0,pi) q[11];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
cx q[2],q[6];
u3(pi/2,0,pi/2) q[3];
u3(0,0,0.02) q[6];
cx q[2],q[6];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,0,pi/2) q[7];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
swap q[2],q[6];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[3];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[4],q[8];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[6];
swap q[6],q[2];
u3(pi/2,-pi/2,pi/2) q[7];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.04) q[2];
cx q[3],q[2];
cx q[2],q[6];
u3(pi/2,0,pi) q[3];
u3(0,0,0.04) q[6];
cx q[2],q[6];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
swap q[2],q[6];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi/2) q[3];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,0,pi/2) q[6];
swap q[6],q[2];
u3(pi/2,0,pi/2) q[7];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
cx q[2],q[6];
u3(pi/2,-pi/2,pi/2) q[3];
u3(0,0,0.02) q[6];
cx q[2],q[6];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[7];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.04) q[2];
cx q[3],q[2];
swap q[2],q[6];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,0,pi/2) q[7];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
swap q[3],q[7];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
u3(pi/2,-pi,-pi) q[12];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
u3(pi/2,0,pi) q[6];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
u3(pi/2,0,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,-pi/2,pi/2) q[3];
u3(pi/2,0,pi) q[13];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
u3(pi/2,0,pi) q[2];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
u3(pi/2,0,pi/2) q[6];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
u3(pi/2,-pi/2,pi/2) q[7];
swap q[7],q[3];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
u3(pi/2,0,pi) q[7];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
swap q[3],q[7];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
u3(pi/2,-pi,-pi) q[14];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,0,pi/2) q[2];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
u3(pi/2,0,pi) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,0,pi/2) q[3];
u3(pi/2,0,pi) q[15];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
u3(pi/2,0,pi/2) q[15];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
u3(pi/2,-pi/2,pi/2) q[15];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
u3(pi/2,0,pi) q[6];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
u3(pi/2,0,pi/2) q[7];
swap q[7],q[3];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
swap q[3],q[7];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
u3(pi/2,0,pi) q[2];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
u3(pi/2,0,pi/2) q[6];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.04) q[7];
cx q[3],q[7];
u3(pi/2,0,pi) q[3];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,0,pi/2) q[2];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
u3(pi/2,0,pi) q[7];
swap q[7],q[3];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
u3(pi/2,0,pi) q[6];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
u3(pi/2,0,pi) q[2];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,0,pi/2) q[7];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi/2,pi/2) q[7];
swap q[7],q[3];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[4],q[8];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,0,pi/2) q[3];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,0,pi/2) q[6];
u3(pi/2,0,pi/2) q[7];
swap q[7],q[3];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,-pi/2,pi/2) q[7];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.04) q[7];
cx q[3],q[7];
u3(pi/2,0,pi) q[3];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
swap q[7],q[3];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,0,pi/2) q[7];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi/2,pi/2) q[7];
swap q[7],q[3];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[4],q[8];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,0,pi/2) q[3];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,0,pi/2) q[6];
swap q[6],q[2];
u3(pi/2,0,pi/2) q[7];
swap q[7],q[3];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
cx q[2],q[6];
u3(pi/2,-pi/2,pi/2) q[3];
u3(0,0,0.02) q[6];
cx q[2],q[6];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[7];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.04) q[2];
cx q[3],q[2];
swap q[2],q[6];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,0,pi) q[3];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[4],q[8];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[7];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,0,pi/2) q[7];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.02) q[7];
cx q[3],q[7];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,-pi/2,pi/2) q[6];
swap q[6],q[2];
u3(pi/2,-pi/2,pi/2) q[7];
swap q[7],q[3];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
u3(pi/2,0,pi) q[10];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
u3(pi/2,0,pi) q[11];
cx q[7],q[3];
u3(0,0,0.04) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.04) q[2];
cx q[3],q[2];
cx q[2],q[6];
u3(pi/2,0,pi) q[3];
u3(0,0,0.04) q[6];
cx q[2],q[6];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,0,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,0,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,0,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
cx q[3],q[2];
u3(0,0,0.02) q[2];
cx q[3],q[2];
swap q[2],q[6];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi/2) q[3];
u3(0,0,0.04) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
cx q[4],q[8];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[2];
u3(pi/2,0,pi/2) q[6];
u3(pi/2,0,pi/2) q[7];
u3(0,0,0.04) q[8];
cx q[4],q[8];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,0,pi/2) q[0];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[13];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[8];
cx q[4],q[8];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,0,pi/2) q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,0,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,0,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,0,pi/2) q[14];
u3(pi/2,0,pi/2) q[15];
u3(pi/2,0,pi/2) q[8];
u3(pi/2,0,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.02) q[9];
cx q[5],q[9];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[9],q[10];
u3(0,0,0.02) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.02) q[11];
cx q[10],q[11];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[11],q[7];
u3(0,0,0.02) q[7];
cx q[11],q[7];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[3];
u3(0,0,0.02) q[3];
cx q[7],q[3];
u3(pi/2,-pi/2,pi/2) q[7];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[5],q[9];
u3(0,0,0.04) q[9];
cx q[5],q[9];
cx q[9],q[10];
u3(0,0,0.04) q[10];
cx q[9],q[10];
cx q[10],q[11];
u3(0,0,0.04) q[11];
cx q[10],q[11];
cx q[11],q[7];
u3(0,0,0.04) q[7];
cx q[11],q[7];
swap q[3],q[7];
cx q[7],q[6];
u3(0,0,0.02) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.02) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.02) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.02) q[0];
cx q[1],q[0];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi/2,-pi/2,pi/2) q[2];
u3(0,0,0.02) q[4];
cx q[0],q[4];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[3],q[7];
u3(0,0,0.04) q[7];
cx q[3],q[7];
cx q[7],q[6];
u3(0,0,0.04) q[6];
cx q[7],q[6];
cx q[6],q[2];
u3(0,0,0.04) q[2];
cx q[6],q[2];
cx q[2],q[1];
u3(0,0,0.04) q[1];
cx q[2],q[1];
cx q[1],q[0];
u3(0,0,0.04) q[0];
cx q[1],q[0];
u3(0,0,0.02) q[8];
cx q[4],q[8];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[0],q[4];
u3(0,0,0.04) q[4];
cx q[0],q[4];
cx q[8],q[12];
u3(0,0,0.02) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.02) q[13];
cx q[12],q[13];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[13],q[14];
u3(0,0,0.02) q[14];
cx q[13],q[14];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[14],q[15];
u3(0,0,0.02) q[15];
cx q[14],q[15];
u3(pi/2,-pi/2,pi/2) q[14];
u3(pi/2,-pi/2,pi/2) q[15];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[4],q[8];
u3(0,0,0.04) q[8];
cx q[4],q[8];
cx q[8],q[12];
u3(0,0,0.04) q[12];
cx q[8],q[12];
cx q[12],q[13];
u3(0,0,0.04) q[13];
cx q[12],q[13];
cx q[13],q[14];
u3(0,0,0.04) q[14];
cx q[13],q[14];
cx q[14],q[15];
u3(0,0,0.04) q[15];
cx q[14],q[15];
