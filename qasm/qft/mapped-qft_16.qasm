OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u3(7*pi/2,4.712341043485068,4*pi) q[10];
cx q[10],q[11];
u3(0,-pi/2,13.351768777756622) q[11];
cx q[10],q[11];
u3(pi/2,1.5707004529956536,13*pi/4) q[11];
cx q[10],q[14];
u3(0,-pi/2,13.744467859455344) q[14];
cx q[10],q[14];
cx q[10],q[6];
u3(0,-pi/2,5*pi/8) q[14];
u3(0,-pi/2,13.940817400304708) q[6];
cx q[10],q[6];
swap q[10],q[11];
cx q[10],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[10],q[14];
u3(pi/2,1.570604579196411,13*pi/4) q[14];
u3(0,-pi/2,9*pi/16) q[6];
cx q[10],q[6];
u3(0,-pi/2,13.744467859455344) q[6];
cx q[10],q[6];
swap q[10],q[14];
u3(0,-pi/2,5*pi/8) q[6];
cx q[10],q[6];
u3(0,-pi/2,13.351768777756622) q[6];
cx q[10],q[6];
u3(pi/2,1.570412831597925,13*pi/4) q[6];
cx q[11],q[15];
u3(0,-pi/2,14.038992170729388) q[15];
cx q[11],q[15];
cx q[11],q[7];
u3(0,-pi/2,1.6689710972195777) q[15];
cx q[14],q[15];
u3(0,-pi/2,13.940817400304708) q[15];
cx q[14],q[15];
u3(0,-pi/2,9*pi/16) q[15];
u3(0,-pi/2,14.088079555941729) q[7];
cx q[11],q[7];
u3(0,-pi/2,1.6198837120072371) q[7];
swap q[7],q[11];
swap q[11],q[10];
cx q[11],q[15];
cx q[14],q[10];
u3(0,-pi/2,14.038992170729388) q[10];
cx q[14],q[10];
u3(0,-pi/2,1.6689710972195777) q[10];
u3(0,-pi/2,13.744467859455344) q[15];
cx q[11],q[15];
cx q[11],q[10];
u3(0,-pi/2,13.940817400304708) q[10];
cx q[11],q[10];
u3(0,-pi/2,9*pi/16) q[10];
u3(0,-pi/2,5*pi/8) q[15];
cx q[7],q[3];
u3(0,-pi/2,14.112623248547898) q[3];
cx q[7],q[3];
u3(0,-pi/2,1.5953400194010667) q[3];
swap q[6],q[7];
cx q[6],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
swap q[5],q[6];
cx q[5],q[4];
u3(0,-pi/2,14.131031018002526) q[4];
cx q[5],q[4];
u3(0,-pi/2,1.576932249946439) q[4];
cx q[5],q[6];
u3(0,-pi/2,14.134098979578297) q[6];
cx q[5],q[6];
cx q[5],q[1];
u3(0,-pi/2,14.135632960366184) q[1];
cx q[5],q[1];
u3(0,-pi/2,1.5723303075827821) q[1];
swap q[4],q[5];
cx q[4],q[0];
u3(0,-pi/2,14.136399950760126) q[0];
cx q[4],q[0];
u3(0,-pi/2,1.5715633171888395) q[0];
cx q[4],q[8];
u3(0,-pi/2,1.573864288370668) q[6];
swap q[7],q[11];
cx q[11],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[11],q[15];
cx q[11],q[10];
u3(0,-pi/2,13.744467859455344) q[10];
cx q[11],q[10];
u3(0,-pi/2,5*pi/8) q[10];
swap q[10],q[14];
u3(pi/2,1.5700293364009537,13*pi/4) q[15];
cx q[15],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[15],q[14];
u3(pi/2,1.569262346007011,13*pi/4) q[14];
swap q[3],q[7];
swap q[6],q[10];
cx q[6],q[7];
u3(0,-pi/2,14.088079555941729) q[7];
cx q[6],q[7];
cx q[6],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
cx q[6],q[5];
u3(0,-pi/2,14.124895094850984) q[5];
cx q[6],q[5];
u3(0,-pi/2,1.5830681730979816) q[5];
cx q[6],q[10];
u3(0,-pi/2,14.131031018002526) q[10];
cx q[6],q[10];
u3(0,-pi/2,1.576932249946439) q[10];
swap q[5],q[6];
cx q[5],q[1];
u3(0,-pi/2,14.134098979578297) q[1];
cx q[5],q[1];
u3(0,-pi/2,1.573864288370668) q[1];
u3(0,-pi/2,1.6198837120072371) q[7];
cx q[3],q[7];
u3(0,-pi/2,14.038992170729388) q[7];
cx q[3],q[7];
cx q[3],q[2];
u3(0,-pi/2,14.088079555941729) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.6198837120072371) q[2];
swap q[2],q[3];
cx q[2],q[6];
u3(0,-pi/2,14.112623248547898) q[6];
cx q[2],q[6];
u3(0,-pi/2,1.5953400194010667) q[6];
u3(0,-pi/2,1.6689710972195777) q[7];
cx q[11],q[7];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[11],q[7];
u3(0,-pi/2,9*pi/16) q[7];
swap q[7],q[11];
cx q[15],q[11];
u3(0,-pi/2,13.744467859455344) q[11];
cx q[15],q[11];
u3(0,-pi/2,5*pi/8) q[11];
swap q[11],q[15];
cx q[14],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[14],q[15];
u3(pi/2,1.5677283652191252,13*pi/4) q[15];
cx q[7],q[3];
u3(0,-pi/2,14.038992170729388) q[3];
cx q[7],q[3];
u3(0,-pi/2,1.6689710972195777) q[3];
cx q[7],q[6];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
swap q[6],q[10];
cx q[2],q[6];
u3(0,-pi/2,14.124895094850984) q[6];
cx q[2],q[6];
cx q[2],q[1];
u3(0,-pi/2,14.131031018002526) q[1];
cx q[2],q[1];
u3(0,-pi/2,1.576932249946439) q[1];
swap q[1],q[2];
swap q[0],q[1];
cx q[5],q[1];
u3(0,-pi/2,14.135632960366184) q[1];
cx q[5],q[1];
u3(0,-pi/2,1.5723303075827821) q[1];
cx q[0],q[1];
u3(0,-pi/2,14.134098979578297) q[1];
cx q[0],q[1];
u3(0,-pi/2,1.573864288370668) q[1];
u3(0,-pi/2,1.5830681730979816) q[6];
cx q[7],q[6];
u3(0,-pi/2,14.112623248547898) q[6];
cx q[7],q[6];
swap q[3],q[7];
cx q[11],q[7];
cx q[3],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
u3(0,-pi/2,1.5953400194010667) q[6];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[11],q[7];
cx q[11],q[10];
u3(0,-pi/2,14.038992170729388) q[10];
cx q[11],q[10];
u3(0,-pi/2,1.6689710972195777) q[10];
swap q[10],q[14];
u3(0,-pi/2,9*pi/16) q[7];
swap q[7],q[11];
cx q[10],q[11];
u3(0,-pi/2,13.744467859455344) q[11];
cx q[10],q[11];
cx q[10],q[14];
u3(0,-pi/2,5*pi/8) q[11];
u3(0,-pi/2,13.940817400304708) q[14];
cx q[10],q[14];
u3(0,-pi/2,9*pi/16) q[14];
cx q[15],q[11];
u3(0,-pi/2,13.351768777756622) q[11];
cx q[15],q[11];
u3(pi/2,1.564660403643354,13*pi/4) q[11];
cx q[15],q[14];
u3(0,-pi/2,13.744467859455344) q[14];
cx q[15],q[14];
swap q[11],q[15];
u3(0,-pi/2,5*pi/8) q[14];
cx q[15],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[15],q[14];
u3(pi/2,1.5585244804918115,13*pi/4) q[14];
cx q[7],q[6];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
cx q[10],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[6],q[7];
cx q[11],q[7];
cx q[6],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
swap q[2],q[6];
swap q[1],q[2];
cx q[10],q[6];
cx q[3],q[2];
u3(0,-pi/2,14.131031018002526) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.576932249946439) q[2];
cx q[1],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[1],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[11],q[7];
u3(0,-pi/2,9*pi/16) q[7];
swap q[11],q[7];
cx q[15],q[11];
u3(0,-pi/2,13.744467859455344) q[11];
cx q[15],q[11];
u3(0,-pi/2,5*pi/8) q[11];
swap q[15],q[11];
cx q[14],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[14],q[15];
u3(pi/2,1.5462526341887264,13*pi/4) q[15];
cx q[7],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[6],q[10];
cx q[11],q[10];
u3(0,-pi/2,13.940817400304708) q[10];
cx q[11],q[10];
u3(0,-pi/2,9*pi/16) q[10];
cx q[14],q[10];
u3(0,-pi/2,13.744467859455344) q[10];
cx q[14],q[10];
u3(0,-pi/2,5*pi/8) q[10];
swap q[10],q[14];
cx q[15],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[15],q[14];
u3(pi/2,1.521708941582556,13*pi/4) q[14];
cx q[6],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
swap q[2],q[3];
cx q[7],q[3];
u3(0,-pi/2,14.088079555941729) q[3];
cx q[7],q[3];
u3(0,-pi/2,1.6198837120072371) q[3];
swap q[3],q[7];
cx q[11],q[7];
u3(0,-pi/2,14.038992170729388) q[7];
cx q[11],q[7];
u3(0,-pi/2,1.6689710972195777) q[7];
swap q[7],q[11];
cx q[10],q[11];
u3(0,-pi/2,13.940817400304708) q[11];
cx q[10],q[11];
u3(0,-pi/2,9*pi/16) q[11];
cx q[15],q[11];
u3(0,-pi/2,13.744467859455344) q[11];
cx q[15],q[11];
u3(0,-pi/2,5*pi/8) q[11];
swap q[11],q[15];
cx q[14],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[14],q[15];
u3(pi/2,1.4726215563702154,13*pi/4) q[15];
u3(0,-pi/2,14.136783445957098) q[8];
cx q[4],q[8];
u3(0,-pi/2,1.571179821991868) q[8];
swap q[8],q[4];
cx q[5],q[4];
u3(0,-pi/2,14.136399950760126) q[4];
cx q[5],q[4];
u3(0,-pi/2,1.5715633171888395) q[4];
cx q[0],q[4];
u3(0,-pi/2,14.135632960366184) q[4];
cx q[0],q[4];
u3(0,-pi/2,1.5723303075827821) q[4];
swap q[4],q[0];
swap q[0],q[1];
cx q[2],q[1];
u3(0,-pi/2,14.134098979578297) q[1];
cx q[2],q[1];
u3(0,-pi/2,1.573864288370668) q[1];
cx q[0],q[1];
u3(0,-pi/2,14.131031018002526) q[1];
cx q[0],q[1];
u3(0,-pi/2,1.576932249946439) q[1];
swap q[2],q[1];
cx q[6],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
cx q[3],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
swap q[6],q[2];
cx q[7],q[6];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
cx q[10],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[7],q[6];
cx q[11],q[7];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[11],q[7];
u3(0,-pi/2,9*pi/16) q[7];
swap q[11],q[7];
swap q[15],q[11];
cx q[14],q[15];
u3(0,-pi/2,13.744467859455344) q[15];
cx q[14],q[15];
u3(0,-pi/2,5*pi/8) q[15];
cx q[11],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[11],q[15];
u3(pi/2,7*pi/16,13*pi/4) q[15];
cx q[8],q[9];
u3(0,-pi/2,14.136975193555584) q[9];
cx q[8],q[9];
cx q[8],q[12];
u3(0,-pi/2,14.137071067354826) q[12];
cx q[8],q[12];
u3(0,-pi/2,1.5708922005941395) q[12];
swap q[8],q[12];
cx q[12],q[13];
u3(0,-pi/2,14.137119004254448) q[13];
cx q[12],q[13];
u3(0,-pi/2,1.570844263694518) q[13];
u3(0,-pi/2,1.5709880743933822) q[9];
cx q[5],q[9];
u3(0,-pi/2,14.136783445957098) q[9];
cx q[5],q[9];
u3(0,-pi/2,1.571179821991868) q[9];
swap q[9],q[5];
cx q[4],q[5];
u3(0,-pi/2,14.136399950760126) q[5];
cx q[4],q[5];
u3(0,-pi/2,1.5715633171888395) q[5];
cx q[1],q[5];
u3(0,-pi/2,14.135632960366184) q[5];
cx q[1],q[5];
u3(0,-pi/2,1.5723303075827821) q[5];
swap q[1],q[5];
cx q[0],q[1];
u3(0,-pi/2,14.134098979578297) q[1];
cx q[0],q[1];
u3(0,-pi/2,1.573864288370668) q[1];
cx q[2],q[1];
u3(0,-pi/2,14.131031018002526) q[1];
cx q[2],q[1];
u3(0,-pi/2,1.576932249946439) q[1];
swap q[2],q[1];
cx q[3],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
cx q[6],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
swap q[6],q[2];
cx q[10],q[6];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
cx q[7],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[10],q[6];
cx q[14],q[10];
u3(0,-pi/2,13.940817400304708) q[10];
cx q[14],q[10];
u3(0,-pi/2,9*pi/16) q[10];
cx q[11],q[10];
u3(0,-pi/2,13.744467859455344) q[10];
cx q[11],q[10];
u3(0,-pi/2,5*pi/8) q[10];
swap q[14],q[10];
cx q[15],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[15],q[14];
u3(pi/2,3*pi/8,13*pi/4) q[14];
cx q[9],q[8];
u3(0,-pi/2,14.136975193555584) q[8];
cx q[9],q[8];
u3(0,-pi/2,1.5709880743933822) q[8];
cx q[4],q[8];
u3(0,-pi/2,14.136783445957098) q[8];
cx q[4],q[8];
u3(0,-pi/2,1.571179821991868) q[8];
swap q[8],q[4];
cx q[5],q[4];
u3(0,-pi/2,14.136399950760126) q[4];
cx q[5],q[4];
u3(0,-pi/2,1.5715633171888395) q[4];
cx q[0],q[4];
u3(0,-pi/2,14.135632960366184) q[4];
cx q[0],q[4];
u3(0,-pi/2,1.5723303075827821) q[4];
swap q[4],q[0];
cx q[1],q[0];
u3(0,-pi/2,14.134098979578297) q[0];
cx q[1],q[0];
u3(0,-pi/2,1.573864288370668) q[0];
cx q[9],q[13];
u3(0,-pi/2,14.137071067354826) q[13];
cx q[9],q[13];
u3(0,-pi/2,1.5708922005941395) q[13];
swap q[9],q[13];
cx q[8],q[9];
u3(0,-pi/2,14.136975193555584) q[9];
cx q[8],q[9];
u3(0,-pi/2,1.5709880743933822) q[9];
cx q[5],q[9];
u3(0,-pi/2,14.136783445957098) q[9];
cx q[5],q[9];
u3(0,-pi/2,1.571179821991868) q[9];
swap q[5],q[9];
cx q[4],q[5];
u3(0,-pi/2,14.136399950760126) q[5];
cx q[4],q[5];
u3(0,-pi/2,1.5715633171888395) q[5];
cx q[1],q[5];
u3(0,-pi/2,14.135632960366184) q[5];
cx q[1],q[5];
swap q[1],q[0];
swap q[2],q[1];
cx q[3],q[2];
u3(0,-pi/2,14.131031018002526) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.576932249946439) q[2];
cx q[1],q[2];
u3(0,-pi/2,14.124895094850984) q[2];
cx q[1],q[2];
u3(0,-pi/2,1.5830681730979816) q[2];
swap q[3],q[7];
u3(0,-pi/2,1.5723303075827821) q[5];
cx q[6],q[2];
u3(0,-pi/2,14.112623248547898) q[2];
cx q[6],q[2];
u3(0,-pi/2,1.5953400194010667) q[2];
cx q[3],q[2];
u3(0,-pi/2,14.088079555941729) q[2];
cx q[3],q[2];
u3(0,-pi/2,1.6198837120072371) q[2];
swap q[6],q[2];
cx q[10],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[7],q[6];
cx q[11],q[7];
cx q[6],q[5];
u3(0,-pi/2,14.134098979578297) q[5];
cx q[6],q[5];
u3(0,-pi/2,1.573864288370668) q[5];
cx q[1],q[5];
u3(0,-pi/2,14.131031018002526) q[5];
cx q[1],q[5];
u3(0,-pi/2,1.576932249946439) q[5];
swap q[6],q[5];
cx q[2],q[6];
u3(0,-pi/2,14.124895094850984) q[6];
cx q[2],q[6];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
swap q[2],q[3];
u3(0,-pi/2,1.5830681730979816) q[6];
cx q[2],q[6];
u3(0,-pi/2,14.112623248547898) q[6];
cx q[2],q[6];
u3(0,-pi/2,1.5953400194010667) q[6];
cx q[10],q[6];
u3(0,-pi/2,14.088079555941729) q[6];
cx q[10],q[6];
u3(0,-pi/2,1.6198837120072371) q[6];
u3(0,-pi/2,13.940817400304708) q[7];
cx q[11],q[7];
u3(0,-pi/2,9*pi/16) q[7];
swap q[11],q[7];
cx q[15],q[11];
u3(0,-pi/2,13.744467859455344) q[11];
cx q[15],q[11];
u3(0,-pi/2,5*pi/8) q[11];
swap q[15],q[11];
cx q[14],q[15];
u3(0,-pi/2,13.351768777756622) q[15];
cx q[14],q[15];
u3(pi/2,pi/4,13*pi/4) q[15];
cx q[7],q[6];
u3(0,-pi/2,14.038992170729388) q[6];
cx q[7],q[6];
u3(0,-pi/2,1.6689710972195777) q[6];
swap q[10],q[6];
cx q[11],q[10];
u3(0,-pi/2,13.940817400304708) q[10];
cx q[11],q[10];
u3(0,-pi/2,9*pi/16) q[10];
cx q[14],q[10];
u3(0,-pi/2,13.744467859455344) q[10];
cx q[14],q[10];
u3(0,-pi/2,5*pi/8) q[10];
swap q[14],q[10];
cx q[15],q[14];
u3(0,-pi/2,13.351768777756622) q[14];
cx q[15],q[14];
u3(pi/2,0,13*pi/4) q[14];
swap q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
swap q[6],q[5];
cx q[2],q[6];
swap q[5],q[4];
cx q[4],q[0];
cx q[0],q[4];
cx q[4],q[0];
cx q[6],q[2];
cx q[2],q[6];
swap q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
swap q[9],q[10];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
