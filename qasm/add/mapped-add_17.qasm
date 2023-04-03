OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
u3(0,0,pi/8) q[1];
u3(0,0,pi/4) q[2];
u3(0,0,pi/512) q[6];
u3(0,0,pi/16) q[7];
u3(0,0,pi/128) q[8];
u3(0,0,pi/16) q[10];
u3(0,0,pi/256) q[11];
u3(pi/2,0,pi) q[12];
u3(0,0,pi/8) q[13];
u3(0,0,pi/32) q[15];
u3(0,0,pi/32) q[16];
u3(0,0,pi/4) q[17];
cx q[17],q[12];
u3(0,0,-pi/4) q[12];
cx q[17],q[12];
u3(0,0,pi/4) q[12];
cx q[13],q[12];
u3(0,0,-pi/8) q[12];
cx q[13],q[12];
u3(0,0,pi/8) q[12];
u3(0,0,pi/4) q[13];
u3(pi/2,0,pi) q[17];
cx q[7],q[12];
u3(0,0,-pi/16) q[12];
cx q[7],q[12];
u3(0,0,pi/16) q[12];
swap q[12],q[17];
cx q[13],q[12];
u3(0,0,-pi/4) q[12];
cx q[13],q[12];
u3(0,0,pi/4) q[12];
u3(pi/2,0,pi) q[13];
cx q[16],q[17];
u3(0,0,-pi/32) q[17];
cx q[16],q[17];
u3(0,0,pi/16) q[16];
u3(0,0,pi/32) q[17];
u3(0,0,pi/8) q[7];
cx q[7],q[12];
u3(0,0,-pi/8) q[12];
cx q[7],q[12];
u3(0,0,pi/8) q[12];
u3(0,0,pi/4) q[7];
u3(0,0,pi/64) q[18];
cx q[18],q[17];
u3(0,0,-pi/64) q[17];
cx q[18],q[17];
u3(0,0,pi/64) q[17];
swap q[17],q[12];
cx q[16],q[17];
u3(0,0,-pi/16) q[17];
cx q[16],q[17];
u3(0,0,pi/8) q[16];
u3(0,0,pi/16) q[17];
u3(0,0,pi/32) q[18];
cx q[18],q[17];
u3(0,0,-pi/32) q[17];
cx q[18],q[17];
u3(0,0,pi/32) q[17];
u3(0,0,pi/16) q[18];
swap q[7],q[12];
cx q[12],q[13];
u3(0,0,-pi/4) q[13];
cx q[12],q[13];
u3(pi/2,0,pi) q[12];
swap q[12],q[17];
u3(0,0,pi/4) q[13];
cx q[8],q[7];
u3(0,0,-pi/128) q[7];
cx q[8],q[7];
u3(0,0,pi/128) q[7];
swap q[7],q[12];
cx q[11],q[12];
u3(0,0,-pi/256) q[12];
cx q[11],q[12];
u3(0,0,pi/128) q[11];
u3(0,0,pi/256) q[12];
swap q[11],q[12];
cx q[6],q[11];
u3(0,0,-pi/512) q[11];
cx q[6],q[11];
u3(0,0,pi/512) q[11];
u3(0,0,pi/256) q[6];
u3(0,0,pi/64) q[8];
cx q[8],q[7];
u3(0,0,-pi/64) q[7];
cx q[8],q[7];
u3(0,0,pi/64) q[7];
cx q[12],q[7];
u3(0,0,-pi/128) q[7];
cx q[12],q[7];
u3(0,0,pi/64) q[12];
u3(0,0,pi/128) q[7];
cx q[6],q[7];
u3(0,0,-pi/256) q[7];
cx q[6],q[7];
u3(0,0,pi/128) q[6];
swap q[6],q[11];
u3(0,0,pi/256) q[7];
swap q[2],q[7];
cx q[7],q[6];
u3(0,0,-pi/4) q[6];
cx q[7],q[6];
u3(0,0,pi/4) q[6];
cx q[1],q[6];
u3(0,0,-pi/8) q[6];
cx q[1],q[6];
u3(0,0,pi/4) q[1];
u3(0,0,pi/8) q[6];
swap q[11],q[6];
cx q[10],q[11];
u3(0,0,-pi/16) q[11];
cx q[10],q[11];
u3(0,0,pi/8) q[10];
u3(0,0,pi/16) q[11];
swap q[11],q[16];
swap q[12],q[11];
cx q[12],q[13];
u3(0,0,-pi/8) q[13];
cx q[12],q[13];
u3(0,0,pi/4) q[12];
cx q[12],q[17];
u3(0,0,pi/8) q[13];
cx q[15],q[16];
u3(0,0,-pi/32) q[16];
cx q[15],q[16];
u3(0,0,pi/16) q[15];
u3(0,0,pi/32) q[16];
u3(0,0,-pi/4) q[17];
cx q[12],q[17];
u3(pi/2,0,pi) q[12];
u3(0,0,pi/4) q[17];
cx q[18],q[13];
u3(0,0,-pi/16) q[13];
cx q[18],q[13];
u3(0,0,pi/16) q[13];
u3(0,0,pi/8) q[18];
cx q[18],q[17];
u3(0,0,-pi/8) q[17];
cx q[18],q[17];
u3(0,0,pi/8) q[17];
u3(0,0,pi/4) q[18];
swap q[5],q[10];
u3(pi/2,-pi,2.3789775) q[7];
cx q[2],q[7];
u3(0,-0.92729522,-2.2142974) q[2];
cx q[1],q[2];
u3(0,0,-pi/4) q[2];
cx q[1],q[2];
u3(0,0,pi/2) q[1];
u3(0,0,pi/4) q[2];
swap q[1],q[2];
swap q[6],q[1];
cx q[5],q[6];
u3(0,0,-pi/8) q[6];
cx q[5],q[6];
u3(0,0,pi/4) q[5];
u3(0,0,pi/8) q[6];
u3(pi/2,0.76261516,0) q[7];
u3(0,0,pi/32) q[8];
cx q[8],q[13];
u3(0,0,-pi/32) q[13];
cx q[8],q[13];
u3(0,0,pi/32) q[13];
swap q[13],q[12];
cx q[11],q[12];
u3(0,0,-pi/64) q[12];
cx q[11],q[12];
u3(0,0,pi/32) q[11];
swap q[11],q[6];
u3(0,0,pi/64) q[12];
cx q[18],q[13];
u3(0,0,-pi/4) q[13];
cx q[18],q[13];
u3(0,0,pi/4) q[13];
u3(pi/2,0,pi) q[18];
swap q[7],q[12];
swap q[12],q[17];
swap q[6],q[7];
cx q[1],q[6];
u3(0,0,-pi/128) q[6];
cx q[1],q[6];
u3(0,0,pi/64) q[1];
swap q[2],q[1];
u3(pi/2,-pi,0.83272486) q[6];
cx q[1],q[6];
u3(0,-3*pi/4,-3*pi/4) q[1];
u3(pi/2,2.3334115,0) q[6];
cx q[5],q[6];
u3(0,0,-pi/4) q[6];
cx q[5],q[6];
u3(0,0,pi/2) q[5];
u3(0,0,pi/4) q[6];
u3(0,0,pi/16) q[8];
swap q[8],q[13];
cx q[13],q[12];
u3(0,0,-pi/16) q[12];
cx q[13],q[12];
u3(0,0,pi/16) q[12];
u3(0,0,pi/8) q[13];
cx q[13],q[8];
cx q[7],q[12];
u3(0,0,-pi/32) q[12];
cx q[7],q[12];
u3(0,0,pi/32) q[12];
u3(0,0,pi/16) q[7];
u3(0,0,-pi/8) q[8];
cx q[13],q[8];
u3(0,0,pi/4) q[13];
cx q[13],q[18];
u3(0,0,-pi/4) q[18];
cx q[13],q[18];
u3(pi/2,0,pi) q[13];
u3(0,0,pi/4) q[18];
swap q[13],q[18];
u3(0,0,pi/8) q[8];
cx q[7],q[8];
u3(0,0,-pi/16) q[8];
cx q[7],q[8];
u3(0,0,pi/8) q[7];
swap q[12],q[7];
cx q[12],q[13];
u3(0,0,-pi/8) q[13];
cx q[12],q[13];
u3(0,0,pi/4) q[12];
u3(0,0,pi/8) q[13];
cx q[2],q[7];
u3(0,0,-pi/64) q[7];
cx q[2],q[7];
u3(0,0,pi/32) q[2];
swap q[3],q[2];
u3(0,0,pi/64) q[7];
u3(0,0,pi/16) q[8];
cx q[3],q[8];
u3(0,0,-pi/32) q[8];
cx q[3],q[8];
u3(0,0,pi/16) q[3];
u3(0,0,pi/32) q[8];
u3(0,0,pi/128) q[20];
u3(0,0,pi/64) q[21];
cx q[21],q[16];
u3(0,0,-pi/64) q[16];
cx q[21],q[16];
u3(0,0,pi/64) q[16];
swap q[16],q[15];
cx q[16],q[11];
u3(0,0,-pi/16) q[11];
cx q[16],q[11];
u3(0,0,pi/16) q[11];
u3(0,0,pi/8) q[16];
swap q[16],q[11];
cx q[11],q[6];
cx q[20],q[15];
u3(0,0,-pi/128) q[15];
cx q[20],q[15];
u3(0,0,pi/128) q[15];
u3(0,0,pi/64) q[20];
swap q[15],q[20];
u3(0,0,pi/32) q[21];
cx q[21],q[16];
u3(0,0,-pi/32) q[16];
cx q[21],q[16];
u3(0,0,pi/32) q[16];
cx q[15],q[16];
u3(0,0,-pi/64) q[16];
cx q[15],q[16];
u3(0,0,pi/32) q[15];
u3(0,0,pi/64) q[16];
u3(0,0,pi/16) q[21];
swap q[16],q[21];
u3(0,0,-pi/8) q[6];
cx q[11],q[6];
u3(0,0,pi/4) q[11];
u3(0,0,pi/8) q[6];
swap q[6],q[11];
cx q[16],q[11];
u3(0,0,-pi/16) q[11];
cx q[16],q[11];
u3(0,0,pi/16) q[11];
u3(0,0,pi/8) q[16];
swap q[11],q[16];
cx q[15],q[16];
u3(0,0,-pi/32) q[16];
cx q[15],q[16];
u3(0,0,pi/16) q[15];
swap q[10],q[15];
u3(0,0,pi/32) q[16];
swap q[7],q[6];
u3(pi/2,-pi,0.80818116) q[6];
cx q[5],q[6];
u3(0,-3*pi/4,-3*pi/4) q[5];
u3(pi/2,2.3334115,0) q[6];
cx q[7],q[6];
u3(0,0,-pi/4) q[6];
cx q[7],q[6];
u3(0,0,pi/4) q[6];
cx q[11],q[6];
u3(0,0,-pi/8) q[6];
cx q[11],q[6];
u3(0,0,pi/4) q[11];
u3(0,0,pi/8) q[6];
swap q[6],q[11];
cx q[10],q[11];
u3(0,0,-pi/16) q[11];
cx q[10],q[11];
u3(0,0,pi/8) q[10];
u3(0,0,pi/16) q[11];
swap q[11],q[10];
u3(0,0,pi/2) q[7];
cx q[7],q[8];
u3(0,0,-pi/2) q[8];
cx q[7],q[8];
u3(0,0,pi/2) q[8];
swap q[7],q[8];
swap q[3],q[8];
cx q[6],q[7];
u3(0,0,-pi/4) q[7];
cx q[6],q[7];
u3(0,0,pi/2) q[6];
u3(0,0,pi/4) q[7];
swap q[7],q[6];
cx q[11],q[6];
u3(0,0,-pi/8) q[6];
cx q[11],q[6];
u3(0,0,pi/4) q[11];
u3(0,0,pi/8) q[6];
cx q[8],q[13];
u3(0,0,-pi/16) q[13];
cx q[8],q[13];
u3(0,0,pi/16) q[13];
swap q[12],q[13];
u3(pi/2,-pi,0.80818116) q[12];
cx q[13],q[18];
u3(0,0,-pi/4) q[18];
cx q[13],q[18];
u3(pi/2,0,pi) q[13];
u3(0,0,pi/4) q[18];
cx q[7],q[12];
u3(pi/2,2.3334115,0) q[12];
cx q[11],q[12];
u3(0,0,-pi/4) q[12];
cx q[11],q[12];
u3(0,0,pi/2) q[11];
u3(0,0,pi/4) q[12];
u3(0,-3*pi/4,-3*pi/4) q[7];
u3(0,0,pi/8) q[8];
swap q[8],q[13];
cx q[13],q[18];
u3(0,0,-pi/8) q[18];
cx q[13],q[18];
u3(0,0,pi/4) q[13];
cx q[13],q[8];
u3(0,0,pi/8) q[18];
u3(0,0,-pi/4) q[8];
cx q[13],q[8];
u3(pi/2,0,pi) q[13];
u3(0,0,pi/4) q[8];
swap q[8],q[7];
u3(0,0,pi/256) q[22];
swap q[21],q[22];
cx q[21],q[20];
u3(0,0,-pi/256) q[20];
cx q[21],q[20];
u3(0,0,pi/256) q[20];
cx q[15],q[20];
u3(0,0,-pi/512) q[20];
cx q[15],q[20];
u3(0,0,pi/256) q[15];
u3(0,0,pi/512) q[20];
u3(0,0,pi/128) q[21];
cx q[21],q[22];
u3(0,0,-pi/128) q[22];
cx q[21],q[22];
u3(0,0,pi/64) q[21];
cx q[21],q[16];
u3(0,0,-pi/64) q[16];
cx q[21],q[16];
u3(0,0,pi/64) q[16];
u3(0,0,pi/32) q[21];
swap q[16],q[21];
swap q[15],q[16];
cx q[15],q[10];
u3(0,0,-pi/32) q[10];
cx q[15],q[10];
u3(0,0,pi/32) q[10];
u3(0,0,pi/16) q[15];
swap q[10],q[15];
swap q[11],q[10];
cx q[11],q[6];
u3(0,0,pi/128) q[22];
swap q[17],q[22];
cx q[16],q[17];
u3(0,0,-pi/256) q[17];
cx q[16],q[17];
u3(0,0,pi/128) q[16];
cx q[16],q[21];
u3(0,0,pi/256) q[17];
swap q[18],q[17];
u3(0,0,-pi/128) q[21];
cx q[16],q[21];
u3(0,0,pi/64) q[16];
cx q[16],q[15];
u3(0,0,-pi/64) q[15];
cx q[16],q[15];
u3(0,0,pi/64) q[15];
u3(0,0,pi/32) q[16];
u3(0,0,pi/128) q[21];
u3(0,0,-pi/16) q[6];
cx q[11],q[6];
u3(0,0,pi/8) q[11];
cx q[11],q[12];
u3(0,0,-pi/8) q[12];
cx q[11],q[12];
u3(0,0,pi/4) q[11];
u3(0,0,pi/8) q[12];
swap q[16],q[11];
u3(0,0,pi/16) q[6];
cx q[11],q[6];
u3(0,0,-pi/32) q[6];
cx q[11],q[6];
u3(0,0,pi/16) q[11];
cx q[11],q[12];
u3(0,0,-pi/16) q[12];
cx q[11],q[12];
u3(0,0,pi/8) q[11];
u3(0,0,pi/16) q[12];
swap q[17],q[12];
swap q[11],q[12];
u3(pi/2,-pi,0.80818116) q[11];
cx q[10],q[11];
u3(0,-3*pi/4,-3*pi/4) q[10];
u3(pi/2,2.3334115,0) q[11];
cx q[16],q[11];
u3(0,0,-pi/4) q[11];
cx q[16],q[11];
u3(0,0,pi/4) q[11];
cx q[12],q[11];
u3(0,0,-pi/8) q[11];
cx q[12],q[11];
u3(0,0,pi/8) q[11];
u3(0,0,pi/4) q[12];
swap q[12],q[7];
swap q[11],q[12];
u3(pi/2,-pi,2.3789775) q[16];
cx q[11],q[16];
u3(0,-0.92729522,-2.2142974) q[11];
swap q[12],q[11];
u3(pi/2,0.76261516,0) q[16];
u3(0,0,pi/32) q[6];
cx q[7],q[12];
u3(0,0,-pi/4) q[12];
cx q[7],q[12];
u3(0,0,pi/4) q[12];
swap q[13],q[12];
u3(pi/2,0,-0.80818116) q[12];
u3(0,0,pi/2) q[7];
cx q[7],q[12];
u3(2.3334115,pi/4,-pi/2) q[12];
cx q[12],q[13];
u3(0,0,pi/4) q[13];
cx q[12],q[13];
u3(0,0,-pi/8) q[12];
cx q[12],q[11];
u3(0,0,pi/8) q[11];
cx q[12],q[11];
u3(0,0,-pi/8) q[11];
u3(0,0,-pi/16) q[12];
cx q[12],q[17];
u3(pi/2,-pi/4,3*pi/4) q[13];
u3(0,0,pi/16) q[17];
cx q[12],q[17];
u3(0,0,-pi/32) q[12];
swap q[12],q[11];
cx q[11],q[6];
cx q[13],q[12];
u3(0,0,pi/4) q[12];
cx q[13],q[12];
u3(pi/2,-pi/4,3*pi/4) q[12];
u3(0,0,-pi/8) q[13];
u3(0,0,-pi/16) q[17];
swap q[18],q[17];
cx q[13],q[18];
u3(0,0,pi/8) q[18];
cx q[13],q[18];
u3(0,0,-pi/16) q[13];
swap q[12],q[13];
u3(0,0,-pi/8) q[18];
cx q[13],q[18];
u3(0,0,pi/4) q[18];
cx q[13],q[18];
u3(0,0,-pi/8) q[13];
u3(pi/2,-pi/4,3*pi/4) q[18];
u3(0,0,pi/32) q[6];
cx q[11],q[6];
u3(0,0,-pi/64) q[11];
swap q[11],q[16];
cx q[16],q[15];
u3(0,0,pi/64) q[15];
cx q[16],q[15];
u3(0,0,-pi/64) q[15];
u3(0,0,-pi/128) q[16];
cx q[16],q[21];
u3(0,0,pi/128) q[21];
cx q[16],q[21];
u3(0,0,-pi/256) q[16];
cx q[16],q[17];
u3(0,0,pi/256) q[17];
cx q[16],q[17];
u3(0,0,-pi/512) q[16];
swap q[16],q[15];
cx q[15],q[20];
u3(0,0,-pi/256) q[17];
u3(0,0,pi/512) q[20];
cx q[15],q[20];
u3(0,0,-pi/512) q[20];
u3(0,0,-pi/128) q[21];
u3(0,0,-pi/32) q[6];
swap q[6],q[11];
cx q[12],q[11];
u3(0,0,pi/16) q[11];
cx q[12],q[11];
u3(0,0,-pi/16) q[11];
u3(0,0,-pi/32) q[12];
swap q[12],q[11];
cx q[11],q[16];
cx q[13],q[12];
u3(0,0,pi/8) q[12];
cx q[13],q[12];
u3(0,0,-pi/8) q[12];
u3(0,0,-pi/16) q[13];
swap q[13],q[12];
u3(0,0,pi/32) q[16];
cx q[11],q[16];
u3(0,0,-pi/64) q[11];
u3(0,0,-pi/32) q[16];
swap q[16],q[11];
cx q[12],q[11];
u3(0,0,pi/16) q[11];
cx q[12],q[11];
u3(0,0,-pi/16) q[11];
u3(0,0,-pi/32) q[12];
swap q[11],q[12];
cx q[16],q[21];
cx q[18],q[13];
u3(0,0,pi/4) q[13];
cx q[18],q[13];
u3(pi/2,-pi/4,3*pi/4) q[13];
u3(0,0,-pi/8) q[18];
u3(0,0,pi/64) q[21];
cx q[16],q[21];
u3(0,0,-pi/128) q[16];
cx q[16],q[17];
u3(0,0,pi/128) q[17];
cx q[16],q[17];
u3(0,0,-pi/256) q[16];
u3(0,0,-pi/128) q[17];
swap q[12],q[17];
cx q[18],q[17];
u3(0,0,pi/8) q[17];
cx q[18],q[17];
u3(0,0,-pi/8) q[17];
u3(0,0,-pi/16) q[18];
swap q[17],q[18];
cx q[13],q[18];
u3(0,0,pi/4) q[18];
cx q[13],q[18];
u3(0,0,-pi/8) q[13];
u3(pi/2,-pi/4,3*pi/4) q[18];
u3(0,0,-pi/64) q[21];
swap q[16],q[21];
cx q[11],q[16];
u3(0,0,pi/32) q[16];
cx q[11],q[16];
u3(0,0,-pi/64) q[11];
cx q[11],q[12];
u3(0,0,pi/64) q[12];
cx q[11],q[12];
u3(0,0,-pi/128) q[11];
u3(0,0,-pi/64) q[12];
u3(0,0,-pi/32) q[16];
cx q[17],q[16];
u3(0,0,pi/16) q[16];
cx q[17],q[16];
u3(0,0,-pi/16) q[16];
swap q[11],q[16];
u3(0,0,-pi/32) q[17];
cx q[17],q[12];
u3(0,0,pi/32) q[12];
cx q[17],q[12];
u3(0,0,-pi/32) q[12];
swap q[13],q[12];
cx q[12],q[11];
u3(0,0,pi/8) q[11];
cx q[12],q[11];
u3(0,0,-pi/8) q[11];
u3(0,0,-pi/16) q[12];
cx q[12],q[13];
u3(0,0,pi/16) q[13];
cx q[12],q[13];
u3(0,0,-pi/32) q[12];
swap q[11],q[12];
u3(0,0,-pi/16) q[13];
u3(0,0,-pi/64) q[17];
cx q[21],q[20];
u3(0,0,pi/256) q[20];
cx q[21],q[20];
u3(0,0,-pi/256) q[20];
swap q[15],q[20];
cx q[16],q[15];
u3(0,0,pi/128) q[15];
cx q[16],q[15];
u3(0,0,-pi/128) q[15];
swap q[16],q[15];
cx q[17],q[16];
u3(0,0,pi/64) q[16];
cx q[17],q[16];
u3(0,0,-pi/64) q[16];
cx q[11],q[16];
u3(0,0,pi/32) q[16];
cx q[11],q[16];
u3(0,0,-pi/32) q[16];
swap q[18],q[17];
cx q[17],q[12];
u3(0,0,pi/4) q[12];
cx q[17],q[12];
u3(pi/2,-pi/4,3*pi/4) q[12];
swap q[12],q[13];
u3(0,0,-pi/8) q[17];
cx q[17],q[12];
u3(0,0,pi/8) q[12];
cx q[17],q[12];
u3(0,0,-pi/8) q[12];
cx q[13],q[12];
u3(0,0,pi/4) q[12];
cx q[13],q[12];
u3(pi/2,-pi/4,3*pi/4) q[12];
u3(0,0,-pi/8) q[13];
u3(0,0,-pi/16) q[17];
cx q[17],q[16];
u3(0,0,pi/16) q[16];
cx q[17],q[16];
u3(0,0,-pi/16) q[16];
swap q[17],q[16];
swap q[17],q[12];
cx q[13],q[12];
u3(0,0,pi/8) q[12];
cx q[13],q[12];
u3(0,0,-pi/8) q[12];
cx q[17],q[12];
u3(0,0,pi/4) q[12];
cx q[17],q[12];
u3(pi/2,0,3*pi/4) q[12];
u3(0,1.210812,-2.7816083) q[7];
