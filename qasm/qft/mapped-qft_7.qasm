OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
u3(7*pi/2,4.6878452877785195,4*pi) q[4];
cx q[4],q[7];
u3(0,-pi/2,13.351768777756622) q[7];
cx q[4],q[7];
cx q[4],q[5];
u3(0,-pi/2,13.744467859455344) q[5];
cx q[4],q[5];
u3(0,-pi/2,5*pi/8) q[5];
u3(pi/2,1.521708941582556,13*pi/4) q[7];
swap q[7],q[8];
cx q[4],q[7];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[4],q[7];
cx q[4],q[1];
u3(0,-pi/2,14.038992170729388) q[1];
cx q[4],q[1];
u3(0,-pi/2,1.6689710972195777) q[1];
swap q[1],q[4];
cx q[1],q[2];
u3(0,-pi/2,14.088079555941729) q[2];
cx q[1],q[2];
u3(0,-pi/2,1.6198837120072371) q[2];
u3(0,-pi/2,9*pi/16) q[7];
cx q[8],q[5];
u3(0,-pi/2,13.351768777756622) q[5];
cx q[8],q[5];
u3(pi/2,1.4726215563702154,13*pi/4) q[5];
cx q[8],q[7];
u3(0,-pi/2,13.744467859455344) q[7];
cx q[8],q[7];
u3(0,-pi/2,5*pi/8) q[7];
swap q[4],q[7];
cx q[5],q[4];
u3(0,-pi/2,13.351768777756622) q[4];
cx q[5],q[4];
u3(pi/2,7*pi/16,13*pi/4) q[4];
cx q[8],q[7];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[8],q[7];
swap q[5],q[8];
cx q[5],q[2];
u3(0,-pi/2,14.038992170729388) q[2];
cx q[5],q[2];
u3(0,-pi/2,1.6689710972195777) q[2];
u3(0,-pi/2,9*pi/16) q[7];
cx q[8],q[7];
u3(0,-pi/2,13.744467859455344) q[7];
cx q[8],q[7];
u3(0,-pi/2,5*pi/8) q[7];
cx q[4],q[7];
u3(0,-pi/2,13.351768777756622) q[7];
cx q[4],q[7];
swap q[3],q[4];
cx q[1],q[4];
u3(0,-pi/2,14.112623248547898) q[4];
cx q[1],q[4];
u3(0,-pi/2,1.5953400194010667) q[4];
cx q[5],q[4];
u3(0,-pi/2,14.088079555941729) q[4];
cx q[5],q[4];
swap q[2],q[5];
u3(0,-pi/2,1.6198837120072371) q[4];
u3(pi/2,3*pi/8,13*pi/4) q[7];
cx q[8],q[5];
u3(0,-pi/2,13.940817400304708) q[5];
cx q[8],q[5];
u3(0,-pi/2,9*pi/16) q[5];
swap q[5],q[4];
cx q[3],q[4];
u3(0,-pi/2,13.744467859455344) q[4];
cx q[3],q[4];
u3(0,-pi/2,5*pi/8) q[4];
cx q[7],q[4];
u3(0,-pi/2,13.351768777756622) q[4];
cx q[7],q[4];
u3(pi/2,pi/4,13*pi/4) q[4];
cx q[8],q[5];
u3(0,-pi/2,14.038992170729388) q[5];
cx q[8],q[5];
u3(0,-pi/2,1.6689710972195777) q[5];
swap q[4],q[5];
cx q[3],q[4];
u3(0,-pi/2,13.940817400304708) q[4];
cx q[3],q[4];
u3(0,-pi/2,9*pi/16) q[4];
cx q[7],q[4];
u3(0,-pi/2,13.744467859455344) q[4];
cx q[7],q[4];
u3(0,-pi/2,5*pi/8) q[4];
cx q[5],q[4];
u3(0,-pi/2,13.351768777756622) q[4];
cx q[5],q[4];
u3(pi/2,0,13*pi/4) q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[5],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];