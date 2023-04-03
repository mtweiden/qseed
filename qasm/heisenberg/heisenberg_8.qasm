OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
u3(pi/2,-pi,-pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(pi/2,-pi,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(pi/2,-pi,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi,-pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0200000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-pi/2,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0200000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.0400000000000000) q[1];
cx q[0],q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0200000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-pi/2,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.0400000000000000) q[2];
cx q[1],q[2];
u3(0,-pi,-pi) q[3];
u3(0,1.406583,-1.406583) q[4];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0200000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.0400000000000000) q[3];
cx q[2],q[3];
u3(0,-pi,-pi) q[4];
u3(0,1.406583,-1.406583) q[5];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,0,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0200000000000000) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.0400000000000000) q[4];
cx q[3],q[4];
u3(0,-pi,-pi) q[5];
u3(0,1.406583,-1.406583) q[6];
u3(pi/2,0,pi) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,0,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0200000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[4],q[5];
u3(0,0,0.0400000000000000) q[5];
cx q[4],q[5];
u3(0,-pi,-pi) q[6];
u3(pi/2,0,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0200000000000000) q[7];
cx q[6],q[7];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[5],q[6];
u3(0,0,0.0400000000000000) q[6];
cx q[5],q[6];
u3(pi/2,-pi/2,pi/2) q[7];
cx q[6],q[7];
u3(0,0,0.0400000000000000) q[7];
cx q[6],q[7];
