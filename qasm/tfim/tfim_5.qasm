OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
u3(pi/2,-pi,-pi) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(pi/2,0,pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(pi/2,0,pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(1.5557963,pi/2,-pi) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(1.5557963,pi/2,-pi) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[0];
u3(0,1.406583,-1.406583) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(1.5557963,pi/2,-pi) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,0,pi/2) q[1];
cx q[0],q[1];
u3(0,0,0.500000000000000) q[1];
cx q[0],q[1];
u3(pi/2,-1.5557963,pi/2) q[0];
u3(0,-pi,-pi) q[1];
u3(0,1.406583,-1.406583) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(1.5557963,pi/2,-pi) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,0,pi/2) q[2];
cx q[1],q[2];
u3(0,0,0.500000000000000) q[2];
cx q[1],q[2];
u3(pi/2,-1.5557963,pi/2) q[1];
u3(0,-pi,-pi) q[2];
u3(0,1.406583,-1.406583) q[3];
u3(1.5557963,pi/2,-pi) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,0,pi/2) q[3];
cx q[2],q[3];
u3(0,0,0.500000000000000) q[3];
cx q[2],q[3];
u3(pi/2,-1.5557963,pi/2) q[2];
u3(0,-pi,-pi) q[3];
u3(pi/2,0,pi/2) q[4];
cx q[3],q[4];
u3(0,0,0.500000000000000) q[4];
cx q[3],q[4];
u3(pi/2,-1.5557963,pi/2) q[3];
u3(pi/2,-1.5557963,pi/2) q[4];
