OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u3(1.480320491587989,0.3974404080908114,pi/2) q[1];
u3(0.323922720848898,1.0223261813950035,pi/2) q[2];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
u3(1.4672249933820412,-1.402234865782837,pi/2) q[6];
u3(0.3465955650671772,-0.6380612155108807,pi/2) q[7];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
barrier q[9],q[5],q[8],q[4];
u3(0,0,2.2104116170367) q[4];
u3(0,0,5.95170395263298) q[5];
u3(0,0,5.91277631994994) q[8];
u3(0,0,2.8177419912671) q[9];
cx q[9],q[5];
u3(0,0,0.574364922231565) q[5];
cx q[9],q[5];
cx q[9],q[8];
u3(0,0,0.641815674474444) q[8];
cx q[9],q[8];
swap q[5],q[9];
cx q[5],q[4];
u3(0,0,7.05698318086496) q[4];
cx q[5],q[4];
cx q[9],q[8];
u3(0,0,0.0613918364115587) q[8];
cx q[9],q[8];
swap q[4],q[8];
cx q[9],q[8];
u3(0,0,0.675024269785162) q[8];
cx q[9],q[8];
cx q[4],q[8];
u3(0,0,0.75429598889069) q[8];
cx q[4],q[8];
barrier q[5],q[9],q[4],q[8];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
barrier q[5],q[9],q[4],q[8];
u3(0,0,5.91277631994994) q[4];
u3(0,0,2.8177419912671) q[5];
u3(0,0,2.2104116170367) q[8];
u3(0,0,5.95170395263298) q[9];
cx q[5],q[9];
u3(0,0,0.574364922231565) q[9];
cx q[5],q[9];
cx q[5],q[4];
u3(0,0,0.641815674474444) q[4];
cx q[5],q[4];
swap q[8],q[4];
cx q[5],q[4];
u3(0,0,7.05698318086496) q[4];
cx q[5],q[4];
u3(2.916659701240052,-1.5458211588253383,pi/2) q[5];
cx q[9],q[8];
u3(0,0,0.0613918364115587) q[8];
cx q[9],q[8];
swap q[9],q[5];
cx q[5],q[4];
u3(0,0,0.675024269785162) q[4];
cx q[5],q[4];
u3(1.9005032705302494,-0.7742176324704002,pi/2) q[5];
cx q[8],q[4];
u3(0,0,0.75429598889069) q[4];
cx q[8],q[4];
u3(0.4268601261916944,0.6033056871462663,pi/2) q[4];
u3(2.5630500046700964,1.160032116403519,pi/2) q[8];
cx q[9],q[5];
cx q[9],q[8];
swap q[5],q[9];
cx q[5],q[4];
cx q[5],q[1];
cx q[9],q[8];
swap q[9],q[5];
cx q[5],q[4];
cx q[5],q[1];
cx q[8],q[4];
u3(0.31733775496494154,-1.1063719105154455,pi/2) q[10];
cx q[9],q[10];
u3(1.8008846067944977,0.24699934037421878,pi/2) q[11];
u3(2.501201418138801,-0.8749766084532906,pi/2) q[12];
u3(2.7846776772873425,0.847696929461474,pi/2) q[13];
cx q[9],q[13];
swap q[5],q[9];
cx q[5],q[6];
u3(1.3895674224656858,1.544184449329519,pi/2) q[5];
swap q[1],q[5];
cx q[9],q[10];
cx q[9],q[13];
swap q[5],q[9];
cx q[5],q[6];
u3(0.7291582008075345,1.379146256795872,pi/2) q[5];
cx q[1],q[5];
cx q[8],q[9];
swap q[9],q[8];
cx q[4],q[8];
swap q[4],q[5];
cx q[9],q[10];
swap q[6],q[10];
cx q[5],q[6];
cx q[9],q[13];
cx q[9],q[10];
u3(2.1148538412238165,0.4999423866328554,pi/2) q[9];
swap q[5],q[9];
cx q[1],q[5];
cx q[4],q[5];
cx q[9],q[13];
cx q[9],q[10];
u3(1.352965866223303,0.5790782035337898,pi/2) q[9];
swap q[9],q[5];
cx q[1],q[5];
cx q[4],q[5];
cx q[9],q[5];
swap q[5],q[6];
swap q[8],q[9];
cx q[9],q[5];
cx q[9],q[13];
cx q[9],q[10];
u3(2.810586057509924,-0.22909183728868632,pi/2) q[9];
swap q[9],q[5];
cx q[1],q[5];
cx q[4],q[5];
cx q[9],q[13];
cx q[9],q[10];
u3(0.6436594201332847,-1.444335558549878,pi/2) q[9];
swap q[5],q[9];
cx q[1],q[5];
cx q[4],q[5];
cx q[8],q[9];
swap q[9],q[10];
cx q[13],q[9];
u3(3.0321173924639546,0.6491152769525401,pi/2) q[13];
cx q[6],q[10];
swap q[10],q[6];
u3(0.35942640075632315,-1.1554032568975374,pi/2) q[9];
swap q[9],q[5];
cx q[8],q[9];
cx q[10],q[9];
swap q[10],q[9];
cx q[6],q[10];
swap q[9],q[13];
swap q[5],q[9];
cx q[1],q[5];
cx q[4],q[5];
swap q[9],q[5];
cx q[1],q[5];
u3(2.3186502277227694,-1.040320521412136,pi/2) q[1];
cx q[4],q[5];
u3(1.269952365679867,-0.43842148904420464,pi/2) q[4];
cx q[8],q[9];
cx q[13],q[9];
swap q[5],q[9];
cx q[6],q[5];
swap q[6],q[10];
cx q[6],q[5];
cx q[8],q[9];
cx q[13],q[9];
cx q[10],q[9];
u3(2.3634299114665644,-0.09860356015542227,-2.230528439076293) q[10];
u3(0.5258944689019885,-0.4589409732158267,pi/2) q[13];
u3(1.5268330383660604,0.9307537634925938,pi/2) q[8];
swap q[8],q[12];
swap q[9],q[5];
cx q[6],q[5];
u3(1.66039841699815,3.1358121120332445,2.389653559197333) q[6];
cx q[10],q[6];
cx q[9],q[5];
u3(2.1546941461192595,1.8729509915006934,-2.9277668108758705) q[5];
u3(0.868742021246001,0.325537016130677,-2.5685303123552536) q[9];
cx q[10],q[9];
swap q[10],q[6];
cx q[10],q[9];
cx q[6],q[5];
cx q[6],q[2];
cx q[6],q[7];
swap q[6],q[10];
cx q[6],q[5];
cx q[6],q[2];
cx q[6],q[7];
cx q[9],q[5];
swap q[10],q[9];
swap q[6],q[10];
cx q[6],q[2];
cx q[6],q[7];
swap q[5],q[6];
cx q[6],q[2];
cx q[6],q[7];
cx q[9],q[8];
swap q[9],q[10];
cx q[10],q[11];
cx q[9],q[8];
swap q[4],q[8];
cx q[5],q[4];
swap q[6],q[5];
cx q[5],q[4];
swap q[7],q[6];
cx q[2],q[6];
swap q[5],q[6];
u3(1.7120876324496193,0.7307216388081526,pi/2) q[14];
cx q[10],q[14];
u3(0.8761462616068911,-0.9081424653773653,pi/2) q[10];
swap q[10],q[9];
cx q[10],q[11];
cx q[10],q[14];
u3(3.016946703901294,-1.5443929125535152,pi/2) q[10];
cx q[7],q[11];
swap q[11],q[7];
cx q[6],q[7];
cx q[9],q[10];
swap q[10],q[14];
cx q[11],q[10];
u3(2.4141759071334716,1.3524419560812868,pi/2) q[11];
cx q[6],q[10];
swap q[11],q[10];
u3(2.433789438407942,1.2270213711699638,pi/2) q[6];
cx q[9],q[10];
cx q[14],q[10];
swap q[14],q[10];
swap q[9],q[5];
cx q[5],q[6];
cx q[10],q[6];
swap q[5],q[4];
swap q[6],q[10];
cx q[14],q[10];
swap q[2],q[6];
cx q[6],q[5];
cx q[6],q[7];
swap q[7],q[11];
cx q[6],q[7];
u3(2.433307203011684,-0.27267257801281497,pi/2) q[6];
cx q[9],q[5];
swap q[10],q[9];
cx q[10],q[11];
swap q[6],q[5];
swap q[10],q[6];
cx q[10],q[11];
cx q[4],q[5];
cx q[6],q[7];
u3(1.9476705966118544,-0.291028699849587,pi/2) q[6];
swap q[5],q[6];
cx q[2],q[6];
cx q[4],q[5];
swap q[6],q[10];
cx q[14],q[10];
cx q[6],q[7];
cx q[11],q[7];
u3(2.9641880616621963,1.087474359729903,pi/2) q[11];
u3(0.7965760009630185,0.8549666182294047,pi/2) q[6];
swap q[5],q[6];
cx q[2],q[6];
cx q[4],q[5];
u3(2.431330014698917,1.3830757258354938,pi/2) q[7];
cx q[9],q[10];
swap q[10],q[11];
swap q[6],q[10];
cx q[14],q[10];
swap q[6],q[5];
cx q[2],q[6];
cx q[4],q[5];
cx q[9],q[10];
cx q[11],q[10];
swap q[10],q[6];
cx q[14],q[10];
cx q[9],q[10];
cx q[11],q[10];
cx q[6],q[10];
swap q[6],q[7];
swap q[5],q[6];
cx q[2],q[6];
cx q[4],q[5];
u3(2.3588432787359537,1.1070301910446627,pi/2) q[4];
swap q[6],q[10];
cx q[14],q[10];
swap q[6],q[5];
cx q[2],q[6];
u3(2.380429783174678,1.4608853761186715,pi/2) q[2];
cx q[9],q[10];
cx q[11],q[10];
swap q[10],q[6];
cx q[14],q[10];
u3(0.7780022399277066,-0.8141415813567234,pi/2) q[14];
cx q[7],q[6];
cx q[5],q[6];
cx q[9],q[10];
cx q[11],q[10];
u3(2.928529727557709,1.3343575382103978,pi/2) q[11];
swap q[6],q[10];
cx q[7],q[6];
cx q[5],q[6];
cx q[10],q[6];
u3(0.9154914398651803,-1.1543130107230097,pi/2) q[10];
u3(0.6318597989532175,-0.9564040831667064,pi/2) q[5];
u3(0.6223591101020225,-0.9357444955270107,pi/2) q[6];
u3(0.8357786856577706,1.3612532381143865,pi/2) q[7];
u3(1.0597309533008035,-1.5116014597485483,pi/2) q[9];
