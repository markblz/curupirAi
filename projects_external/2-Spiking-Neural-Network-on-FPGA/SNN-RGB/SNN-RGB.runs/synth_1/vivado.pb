
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
create_project: 2

00:00:042

00:00:232	
510.2032	
217.539Z17-268h px� 
�
Command: %s
1870*	planAhead2�
�read_checkpoint -auto_incremental -incremental {D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/utils_1/imports/synth_1/snn_rgb.dcp}Z12-2866h px� 
�
;Read reference checkpoint from %s for incremental synthesis3154*	planAhead2l
jD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/utils_1/imports/synth_1/snn_rgb.dcpZ12-5825h px� 
T
-Please ensure there are no constraint changes3725*	planAheadZ12-7989h px� 
i
Command: %s
53*	vivadotcl28
6synth_design -top neuron__behave -part xc7a35tcpg236-1Z4-113h px� 
:
Starting synth_design
149*	vivadotclZ4-321h px� 
z
@Attempting to get a license for feature '%s' and/or device '%s'
308*common2
	Synthesis2	
xc7a35tZ17-347h px� 
j
0Got license for feature '%s' and/or device '%s'
310*common2
	Synthesis2	
xc7a35tZ17-349h px� 
D
Loading part %s157*device2
xc7a35tcpg236-1Z21-403h px� 
o
HMultithreading enabled for synth_design using a maximum of %s processes.4828*oasys2
2Z8-7079h px� 
a
?Launching helper process for spawning children vivado processes4827*oasysZ8-7078h px� 
N
#Helper process launched with PID %s4824*oasys2
23512Z8-7075h px� 
�
%s*synth2{
yStarting RTL Elaboration : Time (s): cpu = 00:00:05 ; elapsed = 00:00:20 . Memory (MB): peak = 1341.168 ; gain = 439.578
h px� 
�
synthesizing module '%s'638*oasys2
neuron2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
298@Z8-638h px� 
�
%done synthesizing module '%s' (%s#%s)256*oasys2
neuron2
02
12c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
298@Z8-256h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_1_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
688@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_2_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
748@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_3_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
808@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_4_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
868@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_5_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
928@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_6_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
988@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_3_4_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
1068@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
tmp_sum_5_6_reg2c
_D:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/sources_1/new/neuron.vhd2
1078@Z8-6014h px� 
n
9Port %s in module %s is either unconnected or has no load4866*oasys2
reset2
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_02
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_12
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_22
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_32
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_42
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_52
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_62
neuronZ8-7129h px� 
�
%s*synth2{
yFinished RTL Elaboration : Time (s): cpu = 00:00:06 ; elapsed = 00:00:24 . Memory (MB): peak = 1448.238 ; gain = 546.648
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:06 ; elapsed = 00:00:25 . Memory (MB): peak = 1448.238 ; gain = 546.648
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 1 : Time (s): cpu = 00:00:06 ; elapsed = 00:00:25 . Memory (MB): peak = 1448.238 ; gain = 546.648
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0052

1448.2382
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
>

Processing XDC Constraints
244*projectZ1-262h px� 
=
Initializing timing engine
348*projectZ1-569h px� 
�
Parsing XDC File [%s]
179*designutils2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc8Z20-179h px� 
�
.Invalid option value '%s' specified for '%s'.
161*common2	
13.47ns2
period2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
28@Z17-161h px�
�
No ports matched '%s'.
584*	planAhead2
clk_o2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
38@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2
	input_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
38@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
create_generated_clock2
-objects [get_ports clk_o]2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
38@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2	
reset_n2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
68@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
*_in*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
68@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2
	input_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
68@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_input_delay2&
$-objects [get_ports {reset_n *_in*}]2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
68@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2
led*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
78@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2

output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
78@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_output_delay2
-clock output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
78@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2	
reset_n2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
98@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
*_in*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
98@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2
	input_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
98@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_input_delay2&
$-objects [get_ports {reset_n *_in*}]2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
98@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2
led*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
108@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2

output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
108@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_output_delay2
-clock output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
108@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2	
reset_n2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
128@Z12-584h px�
�
No ports matched '%s'.
584*	planAhead2
*_in*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
128@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2
	input_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
128@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_input_delay2&
$-objects [get_ports {reset_n *_in*}]2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
128@Z12-4739h px�
�
No ports matched '%s'.
584*	planAhead2
led*2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
138@Z12-584h px�
�
clock '%s' not found.
646*	planAhead2

output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
138@Z12-646h px�
�
&%s:No valid object(s) found for '%s'.
2779*	planAhead2
set_output_delay2
-clock output_clk2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc2
138@Z12-4739h px�
�
Finished Parsing XDC File [%s]
178*designutils2Z
VD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/Spiking_NN_RGB_FPGA/VHDL/snn_rgb.sdc8Z20-178h px� 
H
&Completed Processing XDC Constraints

245*projectZ1-263h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0012

1537.2702
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2"
 Constraint Validation Runtime : 2

00:00:002
00:00:00.4122

1537.2702
0.000Z17-268h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
Finished Constraint Validation : Time (s): cpu = 00:00:13 ; elapsed = 00:00:51 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
D
%s
*synth2,
*Start Loading Part and Timing Information
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Loading part: xc7a35tcpg236-1
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Loading Part and Timing Information : Time (s): cpu = 00:00:13 ; elapsed = 00:00:51 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
%s
*synth20
.Start Applying 'set_property' XDC Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished applying 'set_property' XDC Constraints : Time (s): cpu = 00:00:13 ; elapsed = 00:00:51 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
7
%s
*synth2
Start Preparing Guide Design
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
�The reference checkpoint %s is not suitable for use with incremental synthesis for this design. Please regenerate the checkpoint for this design with -incremental_synth switch in the same Vivado session that synth_design has been run. Synthesis will continue with the default flow4740*oasys2l
jD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.srcs/utils_1/imports/synth_1/snn_rgb.dcpZ8-6895h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2~
|Finished Doing Graph Differ : Time (s): cpu = 00:00:13 ; elapsed = 00:00:52 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Preparing Guide Design : Time (s): cpu = 00:00:13 ; elapsed = 00:00:52 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 2 : Time (s): cpu = 00:00:13 ; elapsed = 00:00:52 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
C
%s
*synth2+
)

Incremental Synthesis Report Summary:

h p
x
� 
<
%s
*synth2$
"1. Incremental synthesis run: no

h p
x
� 
O
%s
*synth27
5   Reason for not running incremental synthesis : 


h p
x
� 
�
�Flow is switching to default flow due to incremental criteria not met. If you would like to alter this behaviour and have the flow terminate instead, please set the following parameter config_implementation {autoIncr.Synth.RejectBehavior Terminate}4868*oasysZ8-7130h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Detailed RTL Component Info : 
h p
x
� 
(
%s
*synth2
+---Adders : 
h p
x
� 
F
%s
*synth2.
,	   2 Input   32 Bit       Adders := 5     
h p
x
� 
+
%s
*synth2
+---Registers : 
h p
x
� 
H
%s
*synth20
.	               32 Bit    Registers := 6     
h p
x
� 
H
%s
*synth20
.	                1 Bit    Registers := 1     
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Finished RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
p
%s
*synth2X
VPart Resources:
DSPs: 90 (col length:60)
BRAMs: 100 (col length: RAMB18 60 RAMB36 30)
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
E
%s
*synth2-
+Start Cross Boundary and Area Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
&Parallel synthesis criteria is not met4829*oasysZ8-7080h px� 
n
9Port %s in module %s is either unconnected or has no load4866*oasys2
reset2
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_02
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_12
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_22
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_32
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_42
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_52
neuronZ8-7129h px� 
m
9Port %s in module %s is either unconnected or has no load4866*oasys2
sp_62
neuronZ8-7129h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Cross Boundary and Area Optimization : Time (s): cpu = 00:00:14 ; elapsed = 00:00:56 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
@
%s
*synth2(
&Start Applying XDC Timing Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Applying XDC Timing Constraints : Time (s): cpu = 00:00:18 ; elapsed = 00:01:06 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
4
%s
*synth2
Start Timing Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2
}Finished Timing Optimization : Time (s): cpu = 00:00:18 ; elapsed = 00:01:07 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
3
%s
*synth2
Start Technology Mapping
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2~
|Finished Technology Mapping : Time (s): cpu = 00:00:18 ; elapsed = 00:01:07 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
-
%s
*synth2
Start IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
?
%s
*synth2'
%Start Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
B
%s
*synth2*
(Finished Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2x
vFinished IO Insertion : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Start Renaming Generated Instances
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Instances : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start Rebuilding User Hierarchy
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Rebuilding User Hierarchy : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Renaming Generated Ports
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Ports : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Start Renaming Generated Nets
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Nets : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Writing Synthesis Report
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
/
%s
*synth2

Report BlackBoxes: 
h p
x
� 
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
8
%s
*synth2 
| |BlackBox name |Instances |
h p
x
� 
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
/
%s*synth2

Report Cell Usage: 
h px� 
0
%s*synth2
+------+-----+------+
h px� 
0
%s*synth2
|      |Cell |Count |
h px� 
0
%s*synth2
+------+-----+------+
h px� 
0
%s*synth2
|1     |OBUF |     1|
h px� 
0
%s*synth2
+------+-----+------+
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Writing Synthesis Report : Time (s): cpu = 00:00:23 ; elapsed = 00:01:15 . Memory (MB): peak = 1537.270 ; gain = 635.680
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
`
%s
*synth2H
FSynthesis finished with 0 errors, 1 critical warnings and 9 warnings.
h p
x
� 
�
%s
*synth2�
Synthesis Optimization Runtime : Time (s): cpu = 00:00:16 ; elapsed = 00:01:05 . Memory (MB): peak = 1537.270 ; gain = 546.648
h p
x
� 
�
%s
*synth2�
�Synthesis Optimization Complete : Time (s): cpu = 00:00:23 ; elapsed = 00:01:16 . Memory (MB): peak = 1537.270 ; gain = 635.680
h p
x
� 
B
 Translating synthesized netlist
350*projectZ1-571h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0422

1537.2702
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
Q
)Pushed %s inverter(s) to %s load pin(s).
98*opt2
02
0Z31-138h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002

00:00:002

1537.2702
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
V
%Synth Design complete | Checksum: %s
562*	vivadotcl2

80b182f5Z4-1430h px� 
C
Releasing license: %s
83*common2
	SynthesisZ17-83h px� 

G%s Infos, %s Warnings, %s Critical Warnings and %s Errors encountered.
28*	vivadotcl2
172
422
92
0Z4-41h px� 
L
%s completed successfully
29*	vivadotcl2
synth_designZ4-42h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
synth_design: 2

00:00:242

00:01:402

1537.2702

1023.141Z17-268h px� 
c
%s6*runtcl2G
ESynthesis results are not added to the cache due to CRITICAL_WARNING
h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Write ShapeDB Complete: 2

00:00:002
00:00:00.1052

1537.2702
0.000Z17-268h px� 
�
 The %s '%s' has been generated.
621*common2

checkpoint2[
YD:/Vivado Projects/Spiking-Neural-Network-on-FPGA/SNN-RGB/SNN-RGB.runs/synth_1/neuron.dcpZ17-1381h px� 
�
%s4*runtcl2d
bExecuting : report_utilization -file neuron_utilization_synth.rpt -pb neuron_utilization_synth.pb
h px� 
\
Exiting %s at %s...
206*common2
Vivado2
Wed Dec 27 00:14:42 2023Z17-206h px� 


End Record