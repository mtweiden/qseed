OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[6];
u3(pi/2,-pi,-pi) q[1];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(pi/2,0,pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(1.5557963,pi/2,-pi) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.5) q[2];
cx q[1],q[2];
u3(pi/2,-1.5557963,pi/2) q[1];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,0,pi/2) q[5];
cx q[2],q[5];
u3(0,0,0.5) q[5];
cx q[2],q[5];
u3(pi/2,-1.5557963,pi/2) q[2];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(1.5557963,pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,0,pi/2) q[4];
cx q[5],q[4];
u3(0,0,0.5) q[4];
cx q[5],q[4];
u3(pi/2,-1.5557963,pi/2) q[5];
u3(1.5557963,pi/2,-pi) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,0,pi/2) q[3];
cx q[4],q[3];
u3(0,0,0.5) q[3];
cx q[4],q[3];
u3(pi/2,-1.5557963,pi/2) q[4];
u3(pi/2,0,pi/2) q[6];
cx q[3],q[6];
u3(0,0,0.5) q[6];
cx q[3],q[6];
u3(pi/2,-1.5557963,pi/2) q[3];
u3(pi/2,-1.5557963,pi/2) q[6];
