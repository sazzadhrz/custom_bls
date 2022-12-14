??+
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.32unknown8??$
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0
?
instance_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_6/gamma
?
2instance_normalization_6/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_6/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_6/beta
?
1instance_normalization_6/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_6/beta*
_output_shapes
:*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:@*
dtype0
?
instance_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_7/gamma
?
2instance_normalization_7/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_7/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_7/beta
?
1instance_normalization_7/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_7/beta*
_output_shapes
:*
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_12/kernel
~
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:?*
dtype0
?
instance_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_8/gamma
?
2instance_normalization_8/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_8/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_8/beta
?
1instance_normalization_8/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_8/beta*
_output_shapes
:*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:?*
dtype0
?
instance_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_9/gamma
?
2instance_normalization_9/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_9/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameinstance_normalization_9/beta
?
1instance_normalization_9/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_9/beta*
_output_shapes
:*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:?*
dtype0
?
instance_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!instance_normalization_10/gamma
?
3instance_normalization_10/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_10/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_10/beta
?
2instance_normalization_10/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_10/beta*
_output_shapes
:*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_15/kernel
~
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
?
instance_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!instance_normalization_11/gamma
?
3instance_normalization_11/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_11/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_11/beta
?
2instance_normalization_11/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_11/beta*
_output_shapes
:*
dtype0
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0
?
instance_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!instance_normalization_12/gamma
?
3instance_normalization_12/gamma/Read/ReadVariableOpReadVariableOpinstance_normalization_12/gamma*
_output_shapes
:*
dtype0
?
instance_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name instance_normalization_12/beta
?
2instance_normalization_12/beta/Read/ReadVariableOpReadVariableOpinstance_normalization_12/beta*
_output_shapes
:*
dtype0
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?d
value?dB?d B?d
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
trainable_variables
regularization_losses
	variables
	keras_api
 
signatures
 
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
g
	+gamma
,beta
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
g
	;gamma
<beta
=trainable_variables
>regularization_losses
?	variables
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
R
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
g
	Kgamma
Lbeta
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
R
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
g
	[gamma
\beta
]trainable_variables
^regularization_losses
_	variables
`	keras_api
R
atrainable_variables
bregularization_losses
c	variables
d	keras_api
h

ekernel
fbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
g
	kgamma
lbeta
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
R
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
R
utrainable_variables
vregularization_losses
w	variables
x	keras_api
h

ykernel
zbias
{trainable_variables
|regularization_losses
}	variables
~	keras_api
l
	gamma
	?beta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
m

?gamma
	?beta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
n
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
!0
"1
+2
,3
14
25
;6
<7
A8
B9
K10
L11
Q12
R13
[14
\15
e16
f17
k18
l19
y20
z21
22
?23
?24
?25
?26
?27
?28
?29
 
?
!0
"1
+2
,3
14
25
;6
<7
A8
B9
K10
L11
Q12
R13
[14
\15
e16
f17
k18
l19
y20
z21
22
?23
?24
?25
?26
?27
?28
?29
?
 ?layer_regularization_losses
?non_trainable_variables
trainable_variables
?layer_metrics
regularization_losses
?layers
?metrics
	variables
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?
 ?layer_regularization_losses
?non_trainable_variables
#trainable_variables
?layer_metrics
$regularization_losses
?layers
?metrics
%	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
'trainable_variables
?layer_metrics
(regularization_losses
?layers
?metrics
)	variables
ig
VARIABLE_VALUEinstance_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
?
 ?layer_regularization_losses
?non_trainable_variables
-trainable_variables
?layer_metrics
.regularization_losses
?layers
?metrics
/	variables
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
?
 ?layer_regularization_losses
?non_trainable_variables
3trainable_variables
?layer_metrics
4regularization_losses
?layers
?metrics
5	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
7trainable_variables
?layer_metrics
8regularization_losses
?layers
?metrics
9	variables
ig
VARIABLE_VALUEinstance_normalization_7/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_7/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
?
 ?layer_regularization_losses
?non_trainable_variables
=trainable_variables
?layer_metrics
>regularization_losses
?layers
?metrics
?	variables
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
?
 ?layer_regularization_losses
?non_trainable_variables
Ctrainable_variables
?layer_metrics
Dregularization_losses
?layers
?metrics
E	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
Gtrainable_variables
?layer_metrics
Hregularization_losses
?layers
?metrics
I	variables
ig
VARIABLE_VALUEinstance_normalization_8/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_8/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
?
 ?layer_regularization_losses
?non_trainable_variables
Mtrainable_variables
?layer_metrics
Nregularization_losses
?layers
?metrics
O	variables
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
?
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layer_metrics
Tregularization_losses
?layers
?metrics
U	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
Wtrainable_variables
?layer_metrics
Xregularization_losses
?layers
?metrics
Y	variables
ig
VARIABLE_VALUEinstance_normalization_9/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEinstance_normalization_9/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 

[0
\1
?
 ?layer_regularization_losses
?non_trainable_variables
]trainable_variables
?layer_metrics
^regularization_losses
?layers
?metrics
_	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
atrainable_variables
?layer_metrics
bregularization_losses
?layers
?metrics
c	variables
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1
 

e0
f1
?
 ?layer_regularization_losses
?non_trainable_variables
gtrainable_variables
?layer_metrics
hregularization_losses
?layers
?metrics
i	variables
jh
VARIABLE_VALUEinstance_normalization_10/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEinstance_normalization_10/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
 

k0
l1
?
 ?layer_regularization_losses
?non_trainable_variables
mtrainable_variables
?layer_metrics
nregularization_losses
?layers
?metrics
o	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
qtrainable_variables
?layer_metrics
rregularization_losses
?layers
?metrics
s	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
utrainable_variables
?layer_metrics
vregularization_losses
?layers
?metrics
w	variables
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
 

y0
z1
?
 ?layer_regularization_losses
?non_trainable_variables
{trainable_variables
?layer_metrics
|regularization_losses
?layers
?metrics
}	variables
ki
VARIABLE_VALUEinstance_normalization_11/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEinstance_normalization_11/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE

0
?1
 

0
?1
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
ki
VARIABLE_VALUEinstance_normalization_12/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEinstance_normalization_12/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
 
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_3Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_10/kernelconv2d_10/biasinstance_normalization_6/gammainstance_normalization_6/betaconv2d_11/kernelconv2d_11/biasinstance_normalization_7/gammainstance_normalization_7/betaconv2d_12/kernelconv2d_12/biasinstance_normalization_8/gammainstance_normalization_8/betaconv2d_13/kernelconv2d_13/biasinstance_normalization_9/gammainstance_normalization_9/betaconv2d_14/kernelconv2d_14/biasinstance_normalization_10/gammainstance_normalization_10/betaconv2d_15/kernelconv2d_15/biasinstance_normalization_11/gammainstance_normalization_11/betaconv2d_16/kernelconv2d_16/biasinstance_normalization_12/gammainstance_normalization_12/betaconv2d_17/kernelconv2d_17/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_3623
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp2instance_normalization_6/gamma/Read/ReadVariableOp1instance_normalization_6/beta/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp2instance_normalization_7/gamma/Read/ReadVariableOp1instance_normalization_7/beta/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp2instance_normalization_8/gamma/Read/ReadVariableOp1instance_normalization_8/beta/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp2instance_normalization_9/gamma/Read/ReadVariableOp1instance_normalization_9/beta/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp3instance_normalization_10/gamma/Read/ReadVariableOp2instance_normalization_10/beta/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp3instance_normalization_11/gamma/Read/ReadVariableOp2instance_normalization_11/beta/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp3instance_normalization_12/gamma/Read/ReadVariableOp2instance_normalization_12/beta/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOpConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_5627
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasinstance_normalization_6/gammainstance_normalization_6/betaconv2d_11/kernelconv2d_11/biasinstance_normalization_7/gammainstance_normalization_7/betaconv2d_12/kernelconv2d_12/biasinstance_normalization_8/gammainstance_normalization_8/betaconv2d_13/kernelconv2d_13/biasinstance_normalization_9/gammainstance_normalization_9/betaconv2d_14/kernelconv2d_14/biasinstance_normalization_10/gammainstance_normalization_10/betaconv2d_15/kernelconv2d_15/biasinstance_normalization_11/gammainstance_normalization_11/betaconv2d_16/kernelconv2d_16/biasinstance_normalization_12/gammainstance_normalization_12/betaconv2d_17/kernelconv2d_17/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_5727??"
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5436

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_8_layer_call_fn_4709

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_16642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5505

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_4454

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5037

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_4641

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:????????? @?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? @?:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
c
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_4295

inputs
identity^
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:??????????? 2
	LeakyReluu
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_10_layer_call_fn_5118

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_28172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_2678

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:????????? ?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? ?:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
H
,__inference_up_sampling2d_layer_call_fn_1854

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_18482
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_12_layer_call_fn_4636

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_25412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_2480

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:?????????@?@2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:?????????@?@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
}
(__inference_conv2d_16_layer_call_fn_5337

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_30172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1975

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4354

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addd
subSubinputsMean:output:0*
T0*1
_output_shapes
:??????????? 2
subk
truedivRealDivsub:z:0add:z:0*
T0*1
_output_shapes
:??????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapel
mulMultruediv:z:0Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1p
add_1AddV2mul:z:0Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?~
?
A__inference_model_3_layer_call_and_return_conditional_losses_3493

inputs
conv2d_10_3406
conv2d_10_3408!
instance_normalization_6_3412!
instance_normalization_6_3414
conv2d_11_3417
conv2d_11_3419!
instance_normalization_7_3423!
instance_normalization_7_3425
conv2d_12_3428
conv2d_12_3430!
instance_normalization_8_3434!
instance_normalization_8_3436
conv2d_13_3439
conv2d_13_3441!
instance_normalization_9_3445!
instance_normalization_9_3447
conv2d_14_3451
conv2d_14_3453"
instance_normalization_10_3456"
instance_normalization_10_3458
conv2d_15_3463
conv2d_15_3465"
instance_normalization_11_3468"
instance_normalization_11_3470
conv2d_16_3475
conv2d_16_3477"
instance_normalization_12_3480"
instance_normalization_12_3482
conv2d_17_3487
conv2d_17_3489
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?1instance_normalization_10/StatefulPartitionedCall?1instance_normalization_11/StatefulPartitionedCall?1instance_normalization_12/StatefulPartitionedCall?0instance_normalization_6/StatefulPartitionedCall?0instance_normalization_7/StatefulPartitionedCall?0instance_normalization_8/StatefulPartitionedCall?0instance_normalization_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_3406conv2d_10_3408*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_23092#
!conv2d_10/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_23302
leaky_re_lu_8/PartitionedCall?
0instance_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0instance_normalization_6_3412instance_normalization_6_3414*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_239122
0instance_normalization_6/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_6/StatefulPartitionedCall:output:0conv2d_11_3417conv2d_11_3419*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_24252#
!conv2d_11/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_24462
leaky_re_lu_9/PartitionedCall?
0instance_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0instance_normalization_7_3423instance_normalization_7_3425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_250722
0instance_normalization_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_7/StatefulPartitionedCall:output:0conv2d_12_3428conv2d_12_3430*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_25412#
!conv2d_12/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_25622 
leaky_re_lu_10/PartitionedCall?
0instance_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0instance_normalization_8_3434instance_normalization_8_3436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_262322
0instance_normalization_8/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_8/StatefulPartitionedCall:output:0conv2d_13_3439conv2d_13_3441*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_26572#
!conv2d_13/StatefulPartitionedCall?
leaky_re_lu_11/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_26782 
leaky_re_lu_11/PartitionedCall?
0instance_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0instance_normalization_9_3445instance_normalization_9_3447*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_273922
0instance_normalization_9/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall9instance_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_18482
up_sampling2d/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_14_3451conv2d_14_3453*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_27752#
!conv2d_14/StatefulPartitionedCall?
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0instance_normalization_10_3456instance_normalization_10_3458*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_284423
1instance_normalization_10/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:09instance_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_28752
concatenate/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_19952!
up_sampling2d_1/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_15_3463conv2d_15_3465*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_28962#
!conv2d_15/StatefulPartitionedCall?
1instance_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0instance_normalization_11_3468instance_normalization_11_3470*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_296523
1instance_normalization_11/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall:instance_normalization_11/StatefulPartitionedCall:output:09instance_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_29962
concatenate_1/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_21422!
up_sampling2d_2/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_16_3475conv2d_16_3477*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_30172#
!conv2d_16/StatefulPartitionedCall?
1instance_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0instance_normalization_12_3480instance_normalization_12_3482*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_308623
1instance_normalization_12/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall:instance_normalization_12/StatefulPartitionedCall:output:09instance_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_31172
concatenate_2/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_22892!
up_sampling2d_3/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_17_3487conv2d_17_3489*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_31382#
!conv2d_17/StatefulPartitionedCall?
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall2^instance_normalization_11/StatefulPartitionedCall2^instance_normalization_12/StatefulPartitionedCall1^instance_normalization_6/StatefulPartitionedCall1^instance_normalization_7/StatefulPartitionedCall1^instance_normalization_8/StatefulPartitionedCall1^instance_normalization_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2f
1instance_normalization_11/StatefulPartitionedCall1instance_normalization_11/StatefulPartitionedCall2f
1instance_normalization_12/StatefulPartitionedCall1instance_normalization_12/StatefulPartitionedCall2d
0instance_normalization_6/StatefulPartitionedCall0instance_normalization_6/StatefulPartitionedCall2d
0instance_normalization_7/StatefulPartitionedCall0instance_normalization_7/StatefulPartitionedCall2d
0instance_normalization_8/StatefulPartitionedCall0instance_normalization_8/StatefulPartitionedCall2d
0instance_normalization_9/StatefulPartitionedCall0instance_normalization_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5259

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_8_layer_call_fn_4781

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_25962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_2817

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addu
subSubinputsMean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
sub|
truedivRealDivsub:z:0add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape}
mulMultruediv:z:0Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_4800

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5214

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5364

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
A__inference_model_3_layer_call_and_return_conditional_losses_4141

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource<
8instance_normalization_6_reshape_readvariableop_resource>
:instance_normalization_6_reshape_1_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource<
8instance_normalization_7_reshape_readvariableop_resource>
:instance_normalization_7_reshape_1_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource<
8instance_normalization_8_reshape_readvariableop_resource>
:instance_normalization_8_reshape_1_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource<
8instance_normalization_9_reshape_readvariableop_resource>
:instance_normalization_9_reshape_1_readvariableop_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource=
9instance_normalization_10_reshape_readvariableop_resource?
;instance_normalization_10_reshape_1_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource=
9instance_normalization_11_reshape_readvariableop_resource?
;instance_normalization_11_reshape_1_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource=
9instance_normalization_12_reshape_readvariableop_resource?
;instance_normalization_12_reshape_1_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?0instance_normalization_10/Reshape/ReadVariableOp?2instance_normalization_10/Reshape_1/ReadVariableOp?0instance_normalization_11/Reshape/ReadVariableOp?2instance_normalization_11/Reshape_1/ReadVariableOp?0instance_normalization_12/Reshape/ReadVariableOp?2instance_normalization_12/Reshape_1/ReadVariableOp?/instance_normalization_6/Reshape/ReadVariableOp?1instance_normalization_6/Reshape_1/ReadVariableOp?/instance_normalization_7/Reshape/ReadVariableOp?1instance_normalization_7/Reshape_1/ReadVariableOp?/instance_normalization_8/Reshape/ReadVariableOp?1instance_normalization_8/Reshape_1/ReadVariableOp?/instance_normalization_9/Reshape/ReadVariableOp?1instance_normalization_9/Reshape_1/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_10/BiasAdd?
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_10/BiasAdd:output:0*1
_output_shapes
:??????????? 2
leaky_re_lu_8/LeakyRelu?
/instance_normalization_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_6/Mean/reduction_indices?
instance_normalization_6/MeanMean%leaky_re_lu_8/LeakyRelu:activations:08instance_normalization_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_6/Mean?
Jinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_6/reduce_std/reduce_variance/MeanMean%leaky_re_lu_8/LeakyRelu:activations:0Sinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_6/reduce_std/reduce_variance/Mean?
7instance_normalization_6/reduce_std/reduce_variance/subSub%leaky_re_lu_8/LeakyRelu:activations:0Ainstance_normalization_6/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 29
7instance_normalization_6/reduce_std/reduce_variance/sub?
:instance_normalization_6/reduce_std/reduce_variance/SquareSquare;instance_normalization_6/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2<
:instance_normalization_6/reduce_std/reduce_variance/Square?
Linstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_6/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_6/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_6/reduce_std/reduce_variance/Mean_1?
(instance_normalization_6/reduce_std/SqrtSqrtCinstance_normalization_6/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_6/reduce_std/Sqrt?
instance_normalization_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_6/add/y?
instance_normalization_6/addAddV2,instance_normalization_6/reduce_std/Sqrt:y:0'instance_normalization_6/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_6/add?
instance_normalization_6/subSub%leaky_re_lu_8/LeakyRelu:activations:0&instance_normalization_6/Mean:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_6/sub?
 instance_normalization_6/truedivRealDiv instance_normalization_6/sub:z:0 instance_normalization_6/add:z:0*
T0*1
_output_shapes
:??????????? 2"
 instance_normalization_6/truediv?
/instance_normalization_6/Reshape/ReadVariableOpReadVariableOp8instance_normalization_6_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_6/Reshape/ReadVariableOp?
&instance_normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_6/Reshape/shape?
 instance_normalization_6/ReshapeReshape7instance_normalization_6/Reshape/ReadVariableOp:value:0/instance_normalization_6/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_6/Reshape?
instance_normalization_6/mulMul$instance_normalization_6/truediv:z:0)instance_normalization_6/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_6/mul?
1instance_normalization_6/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_6_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_6/Reshape_1/ReadVariableOp?
(instance_normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_6/Reshape_1/shape?
"instance_normalization_6/Reshape_1Reshape9instance_normalization_6/Reshape_1/ReadVariableOp:value:01instance_normalization_6/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_6/Reshape_1?
instance_normalization_6/add_1AddV2 instance_normalization_6/mul:z:0+instance_normalization_6/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2 
instance_normalization_6/add_1?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D"instance_normalization_6/add_1:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_11/BiasAdd?
leaky_re_lu_9/LeakyRelu	LeakyReluconv2d_11/BiasAdd:output:0*0
_output_shapes
:?????????@?@2
leaky_re_lu_9/LeakyRelu?
/instance_normalization_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_7/Mean/reduction_indices?
instance_normalization_7/MeanMean%leaky_re_lu_9/LeakyRelu:activations:08instance_normalization_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_7/Mean?
Jinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_7/reduce_std/reduce_variance/MeanMean%leaky_re_lu_9/LeakyRelu:activations:0Sinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_7/reduce_std/reduce_variance/Mean?
7instance_normalization_7/reduce_std/reduce_variance/subSub%leaky_re_lu_9/LeakyRelu:activations:0Ainstance_normalization_7/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@29
7instance_normalization_7/reduce_std/reduce_variance/sub?
:instance_normalization_7/reduce_std/reduce_variance/SquareSquare;instance_normalization_7/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2<
:instance_normalization_7/reduce_std/reduce_variance/Square?
Linstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_7/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_7/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_7/reduce_std/reduce_variance/Mean_1?
(instance_normalization_7/reduce_std/SqrtSqrtCinstance_normalization_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_7/reduce_std/Sqrt?
instance_normalization_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_7/add/y?
instance_normalization_7/addAddV2,instance_normalization_7/reduce_std/Sqrt:y:0'instance_normalization_7/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_7/add?
instance_normalization_7/subSub%leaky_re_lu_9/LeakyRelu:activations:0&instance_normalization_7/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_7/sub?
 instance_normalization_7/truedivRealDiv instance_normalization_7/sub:z:0 instance_normalization_7/add:z:0*
T0*0
_output_shapes
:?????????@?@2"
 instance_normalization_7/truediv?
/instance_normalization_7/Reshape/ReadVariableOpReadVariableOp8instance_normalization_7_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_7/Reshape/ReadVariableOp?
&instance_normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_7/Reshape/shape?
 instance_normalization_7/ReshapeReshape7instance_normalization_7/Reshape/ReadVariableOp:value:0/instance_normalization_7/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_7/Reshape?
instance_normalization_7/mulMul$instance_normalization_7/truediv:z:0)instance_normalization_7/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_7/mul?
1instance_normalization_7/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_7_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_7/Reshape_1/ReadVariableOp?
(instance_normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_7/Reshape_1/shape?
"instance_normalization_7/Reshape_1Reshape9instance_normalization_7/Reshape_1/ReadVariableOp:value:01instance_normalization_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_7/Reshape_1?
instance_normalization_7/add_1AddV2 instance_normalization_7/mul:z:0+instance_normalization_7/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2 
instance_normalization_7/add_1?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D"instance_normalization_7/add_1:z:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
conv2d_12/BiasAdd?
leaky_re_lu_10/LeakyRelu	LeakyReluconv2d_12/BiasAdd:output:0*0
_output_shapes
:????????? @?2
leaky_re_lu_10/LeakyRelu?
/instance_normalization_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_8/Mean/reduction_indices?
instance_normalization_8/MeanMean&leaky_re_lu_10/LeakyRelu:activations:08instance_normalization_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_8/Mean?
Jinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_8/reduce_std/reduce_variance/MeanMean&leaky_re_lu_10/LeakyRelu:activations:0Sinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_8/reduce_std/reduce_variance/Mean?
7instance_normalization_8/reduce_std/reduce_variance/subSub&leaky_re_lu_10/LeakyRelu:activations:0Ainstance_normalization_8/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?29
7instance_normalization_8/reduce_std/reduce_variance/sub?
:instance_normalization_8/reduce_std/reduce_variance/SquareSquare;instance_normalization_8/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2<
:instance_normalization_8/reduce_std/reduce_variance/Square?
Linstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_8/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_8/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_8/reduce_std/reduce_variance/Mean_1?
(instance_normalization_8/reduce_std/SqrtSqrtCinstance_normalization_8/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_8/reduce_std/Sqrt?
instance_normalization_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_8/add/y?
instance_normalization_8/addAddV2,instance_normalization_8/reduce_std/Sqrt:y:0'instance_normalization_8/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_8/add?
instance_normalization_8/subSub&leaky_re_lu_10/LeakyRelu:activations:0&instance_normalization_8/Mean:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_8/sub?
 instance_normalization_8/truedivRealDiv instance_normalization_8/sub:z:0 instance_normalization_8/add:z:0*
T0*0
_output_shapes
:????????? @?2"
 instance_normalization_8/truediv?
/instance_normalization_8/Reshape/ReadVariableOpReadVariableOp8instance_normalization_8_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_8/Reshape/ReadVariableOp?
&instance_normalization_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_8/Reshape/shape?
 instance_normalization_8/ReshapeReshape7instance_normalization_8/Reshape/ReadVariableOp:value:0/instance_normalization_8/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_8/Reshape?
instance_normalization_8/mulMul$instance_normalization_8/truediv:z:0)instance_normalization_8/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_8/mul?
1instance_normalization_8/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_8_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_8/Reshape_1/ReadVariableOp?
(instance_normalization_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_8/Reshape_1/shape?
"instance_normalization_8/Reshape_1Reshape9instance_normalization_8/Reshape_1/ReadVariableOp:value:01instance_normalization_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_8/Reshape_1?
instance_normalization_8/add_1AddV2 instance_normalization_8/mul:z:0+instance_normalization_8/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2 
instance_normalization_8/add_1?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D"instance_normalization_8/add_1:z:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
conv2d_13/BiasAdd?
leaky_re_lu_11/LeakyRelu	LeakyReluconv2d_13/BiasAdd:output:0*0
_output_shapes
:????????? ?2
leaky_re_lu_11/LeakyRelu?
/instance_normalization_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_9/Mean/reduction_indices?
instance_normalization_9/MeanMean&leaky_re_lu_11/LeakyRelu:activations:08instance_normalization_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_9/Mean?
Jinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_9/reduce_std/reduce_variance/MeanMean&leaky_re_lu_11/LeakyRelu:activations:0Sinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_9/reduce_std/reduce_variance/Mean?
7instance_normalization_9/reduce_std/reduce_variance/subSub&leaky_re_lu_11/LeakyRelu:activations:0Ainstance_normalization_9/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?29
7instance_normalization_9/reduce_std/reduce_variance/sub?
:instance_normalization_9/reduce_std/reduce_variance/SquareSquare;instance_normalization_9/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2<
:instance_normalization_9/reduce_std/reduce_variance/Square?
Linstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_9/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_9/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_9/reduce_std/reduce_variance/Mean_1?
(instance_normalization_9/reduce_std/SqrtSqrtCinstance_normalization_9/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_9/reduce_std/Sqrt?
instance_normalization_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_9/add/y?
instance_normalization_9/addAddV2,instance_normalization_9/reduce_std/Sqrt:y:0'instance_normalization_9/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_9/add?
instance_normalization_9/subSub&leaky_re_lu_11/LeakyRelu:activations:0&instance_normalization_9/Mean:output:0*
T0*0
_output_shapes
:????????? ?2
instance_normalization_9/sub?
 instance_normalization_9/truedivRealDiv instance_normalization_9/sub:z:0 instance_normalization_9/add:z:0*
T0*0
_output_shapes
:????????? ?2"
 instance_normalization_9/truediv?
/instance_normalization_9/Reshape/ReadVariableOpReadVariableOp8instance_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_9/Reshape/ReadVariableOp?
&instance_normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_9/Reshape/shape?
 instance_normalization_9/ReshapeReshape7instance_normalization_9/Reshape/ReadVariableOp:value:0/instance_normalization_9/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_9/Reshape?
instance_normalization_9/mulMul$instance_normalization_9/truediv:z:0)instance_normalization_9/Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
instance_normalization_9/mul?
1instance_normalization_9/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_9/Reshape_1/ReadVariableOp?
(instance_normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_9/Reshape_1/shape?
"instance_normalization_9/Reshape_1Reshape9instance_normalization_9/Reshape_1/ReadVariableOp:value:01instance_normalization_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_9/Reshape_1?
instance_normalization_9/add_1AddV2 instance_normalization_9/mul:z:0+instance_normalization_9/Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2 
instance_normalization_9/add_1|
up_sampling2d/ShapeShape"instance_normalization_9/add_1:z:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor"instance_normalization_9/add_1:z:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:????????? @?*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:????????? @?2
conv2d_14/Relu?
0instance_normalization_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_10/Mean/reduction_indices?
instance_normalization_10/MeanMeanconv2d_14/Relu:activations:09instance_normalization_10/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_10/Mean?
Kinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_10/reduce_std/reduce_variance/MeanMeanconv2d_14/Relu:activations:0Tinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_10/reduce_std/reduce_variance/Mean?
8instance_normalization_10/reduce_std/reduce_variance/subSubconv2d_14/Relu:activations:0Binstance_normalization_10/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2:
8instance_normalization_10/reduce_std/reduce_variance/sub?
;instance_normalization_10/reduce_std/reduce_variance/SquareSquare<instance_normalization_10/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2=
;instance_normalization_10/reduce_std/reduce_variance/Square?
Minstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_10/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_10/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_10/reduce_std/reduce_variance/Mean_1?
)instance_normalization_10/reduce_std/SqrtSqrtDinstance_normalization_10/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_10/reduce_std/Sqrt?
instance_normalization_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_10/add/y?
instance_normalization_10/addAddV2-instance_normalization_10/reduce_std/Sqrt:y:0(instance_normalization_10/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_10/add?
instance_normalization_10/subSubconv2d_14/Relu:activations:0'instance_normalization_10/Mean:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_10/sub?
!instance_normalization_10/truedivRealDiv!instance_normalization_10/sub:z:0!instance_normalization_10/add:z:0*
T0*0
_output_shapes
:????????? @?2#
!instance_normalization_10/truediv?
0instance_normalization_10/Reshape/ReadVariableOpReadVariableOp9instance_normalization_10_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_10/Reshape/ReadVariableOp?
'instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_10/Reshape/shape?
!instance_normalization_10/ReshapeReshape8instance_normalization_10/Reshape/ReadVariableOp:value:00instance_normalization_10/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_10/Reshape?
instance_normalization_10/mulMul%instance_normalization_10/truediv:z:0*instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_10/mul?
2instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_10/Reshape_1/ReadVariableOp?
)instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_10/Reshape_1/shape?
#instance_normalization_10/Reshape_1Reshape:instance_normalization_10/Reshape_1/ReadVariableOp:value:02instance_normalization_10/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_10/Reshape_1?
instance_normalization_10/add_1AddV2!instance_normalization_10/mul:z:0,instance_normalization_10/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2!
instance_normalization_10/add_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2#instance_normalization_10/add_1:z:0"instance_normalization_8/add_1:z:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:????????? @?2
concatenate/concaty
up_sampling2d_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:?????????@??*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_15/BiasAdd
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_15/Relu?
0instance_normalization_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_11/Mean/reduction_indices?
instance_normalization_11/MeanMeanconv2d_15/Relu:activations:09instance_normalization_11/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_11/Mean?
Kinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_11/reduce_std/reduce_variance/MeanMeanconv2d_15/Relu:activations:0Tinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_11/reduce_std/reduce_variance/Mean?
8instance_normalization_11/reduce_std/reduce_variance/subSubconv2d_15/Relu:activations:0Binstance_normalization_11/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2:
8instance_normalization_11/reduce_std/reduce_variance/sub?
;instance_normalization_11/reduce_std/reduce_variance/SquareSquare<instance_normalization_11/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2=
;instance_normalization_11/reduce_std/reduce_variance/Square?
Minstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_11/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_11/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_11/reduce_std/reduce_variance/Mean_1?
)instance_normalization_11/reduce_std/SqrtSqrtDinstance_normalization_11/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_11/reduce_std/Sqrt?
instance_normalization_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_11/add/y?
instance_normalization_11/addAddV2-instance_normalization_11/reduce_std/Sqrt:y:0(instance_normalization_11/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_11/add?
instance_normalization_11/subSubconv2d_15/Relu:activations:0'instance_normalization_11/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_11/sub?
!instance_normalization_11/truedivRealDiv!instance_normalization_11/sub:z:0!instance_normalization_11/add:z:0*
T0*0
_output_shapes
:?????????@?@2#
!instance_normalization_11/truediv?
0instance_normalization_11/Reshape/ReadVariableOpReadVariableOp9instance_normalization_11_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_11/Reshape/ReadVariableOp?
'instance_normalization_11/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_11/Reshape/shape?
!instance_normalization_11/ReshapeReshape8instance_normalization_11/Reshape/ReadVariableOp:value:00instance_normalization_11/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_11/Reshape?
instance_normalization_11/mulMul%instance_normalization_11/truediv:z:0*instance_normalization_11/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_11/mul?
2instance_normalization_11/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_11_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_11/Reshape_1/ReadVariableOp?
)instance_normalization_11/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_11/Reshape_1/shape?
#instance_normalization_11/Reshape_1Reshape:instance_normalization_11/Reshape_1/ReadVariableOp:value:02instance_normalization_11/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_11/Reshape_1?
instance_normalization_11/add_1AddV2!instance_normalization_11/mul:z:0,instance_normalization_11/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2!
instance_normalization_11/add_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2#instance_normalization_11/add_1:z:0"instance_normalization_7/add_1:z:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????@??2
concatenate_1/concat{
up_sampling2d_2/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape?
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack?
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1?
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/BiasAdd?
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/Relu?
0instance_normalization_12/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_12/Mean/reduction_indices?
instance_normalization_12/MeanMeanconv2d_16/Relu:activations:09instance_normalization_12/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_12/Mean?
Kinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_12/reduce_std/reduce_variance/MeanMeanconv2d_16/Relu:activations:0Tinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_12/reduce_std/reduce_variance/Mean?
8instance_normalization_12/reduce_std/reduce_variance/subSubconv2d_16/Relu:activations:0Binstance_normalization_12/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2:
8instance_normalization_12/reduce_std/reduce_variance/sub?
;instance_normalization_12/reduce_std/reduce_variance/SquareSquare<instance_normalization_12/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2=
;instance_normalization_12/reduce_std/reduce_variance/Square?
Minstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_12/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_12/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_12/reduce_std/reduce_variance/Mean_1?
)instance_normalization_12/reduce_std/SqrtSqrtDinstance_normalization_12/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_12/reduce_std/Sqrt?
instance_normalization_12/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_12/add/y?
instance_normalization_12/addAddV2-instance_normalization_12/reduce_std/Sqrt:y:0(instance_normalization_12/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_12/add?
instance_normalization_12/subSubconv2d_16/Relu:activations:0'instance_normalization_12/Mean:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_12/sub?
!instance_normalization_12/truedivRealDiv!instance_normalization_12/sub:z:0!instance_normalization_12/add:z:0*
T0*1
_output_shapes
:??????????? 2#
!instance_normalization_12/truediv?
0instance_normalization_12/Reshape/ReadVariableOpReadVariableOp9instance_normalization_12_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_12/Reshape/ReadVariableOp?
'instance_normalization_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_12/Reshape/shape?
!instance_normalization_12/ReshapeReshape8instance_normalization_12/Reshape/ReadVariableOp:value:00instance_normalization_12/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_12/Reshape?
instance_normalization_12/mulMul%instance_normalization_12/truediv:z:0*instance_normalization_12/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_12/mul?
2instance_normalization_12/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_12_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_12/Reshape_1/ReadVariableOp?
)instance_normalization_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_12/Reshape_1/shape?
#instance_normalization_12/Reshape_1Reshape:instance_normalization_12/Reshape_1/ReadVariableOp:value:02instance_normalization_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_12/Reshape_1?
instance_normalization_12/add_1AddV2!instance_normalization_12/mul:z:0,instance_normalization_12/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2!
instance_normalization_12/add_1x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2#instance_normalization_12/add_1:z:0"instance_normalization_6/add_1:z:0"concatenate_2/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_2/concat{
up_sampling2d_3/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape?
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack?
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1?
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2?
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const?
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_2/concat:output:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_17/BiasAdd?
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_17/Tanh?

IdentityIdentityconv2d_17/Tanh:y:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp1^instance_normalization_10/Reshape/ReadVariableOp3^instance_normalization_10/Reshape_1/ReadVariableOp1^instance_normalization_11/Reshape/ReadVariableOp3^instance_normalization_11/Reshape_1/ReadVariableOp1^instance_normalization_12/Reshape/ReadVariableOp3^instance_normalization_12/Reshape_1/ReadVariableOp0^instance_normalization_6/Reshape/ReadVariableOp2^instance_normalization_6/Reshape_1/ReadVariableOp0^instance_normalization_7/Reshape/ReadVariableOp2^instance_normalization_7/Reshape_1/ReadVariableOp0^instance_normalization_8/Reshape/ReadVariableOp2^instance_normalization_8/Reshape_1/ReadVariableOp0^instance_normalization_9/Reshape/ReadVariableOp2^instance_normalization_9/Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2d
0instance_normalization_10/Reshape/ReadVariableOp0instance_normalization_10/Reshape/ReadVariableOp2h
2instance_normalization_10/Reshape_1/ReadVariableOp2instance_normalization_10/Reshape_1/ReadVariableOp2d
0instance_normalization_11/Reshape/ReadVariableOp0instance_normalization_11/Reshape/ReadVariableOp2h
2instance_normalization_11/Reshape_1/ReadVariableOp2instance_normalization_11/Reshape_1/ReadVariableOp2d
0instance_normalization_12/Reshape/ReadVariableOp0instance_normalization_12/Reshape/ReadVariableOp2h
2instance_normalization_12/Reshape_1/ReadVariableOp2instance_normalization_12/Reshape_1/ReadVariableOp2b
/instance_normalization_6/Reshape/ReadVariableOp/instance_normalization_6/Reshape/ReadVariableOp2f
1instance_normalization_6/Reshape_1/ReadVariableOp1instance_normalization_6/Reshape_1/ReadVariableOp2b
/instance_normalization_7/Reshape/ReadVariableOp/instance_normalization_7/Reshape/ReadVariableOp2f
1instance_normalization_7/Reshape_1/ReadVariableOp1instance_normalization_7/Reshape_1/ReadVariableOp2b
/instance_normalization_8/Reshape/ReadVariableOp/instance_normalization_8/Reshape/ReadVariableOp2f
1instance_normalization_8/Reshape_1/ReadVariableOp1instance_normalization_8/Reshape_1/ReadVariableOp2b
/instance_normalization_9/Reshape/ReadVariableOp/instance_normalization_9/Reshape/ReadVariableOp2f
1instance_normalization_9/Reshape_1/ReadVariableOp1instance_normalization_9/Reshape_1/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_2896

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5082

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addu
subSubinputsMean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
sub|
truedivRealDivsub:z:0add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape}
mulMultruediv:z:0Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
X
,__inference_concatenate_2_layer_call_fn_5494
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_31172
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_2965

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_6_layer_call_fn_4363

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_23642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_11_layer_call_fn_5232

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_21222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4918

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_8_layer_call_fn_4790

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_26232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_2086

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5109

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addu
subSubinputsMean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
sub|
truedivRealDivsub:z:0add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape}
mulMultruediv:z:0Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5187

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4500

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_3_layer_call_fn_4206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_33382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4327

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addd
subSubinputsMean:output:0*
T0*1
_output_shapes
:??????????? 2
subk
truedivRealDivsub:z:0add:z:0*
T0*1
_output_shapes
:??????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapel
mulMultruediv:z:0Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1p
add_1AddV2mul:z:0Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4846

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? ?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? ?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_5151

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
s
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5488
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
e
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2289

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_4974

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_15_layer_call_fn_5160

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_28962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
X
,__inference_concatenate_1_layer_call_fn_5317
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_29962
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????@??2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????@:?????????@?@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????@?@
"
_user_specified_name
inputs/1
?
?
 __inference__traced_restore_5727
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias5
1assignvariableop_2_instance_normalization_6_gamma4
0assignvariableop_3_instance_normalization_6_beta'
#assignvariableop_4_conv2d_11_kernel%
!assignvariableop_5_conv2d_11_bias5
1assignvariableop_6_instance_normalization_7_gamma4
0assignvariableop_7_instance_normalization_7_beta'
#assignvariableop_8_conv2d_12_kernel%
!assignvariableop_9_conv2d_12_bias6
2assignvariableop_10_instance_normalization_8_gamma5
1assignvariableop_11_instance_normalization_8_beta(
$assignvariableop_12_conv2d_13_kernel&
"assignvariableop_13_conv2d_13_bias6
2assignvariableop_14_instance_normalization_9_gamma5
1assignvariableop_15_instance_normalization_9_beta(
$assignvariableop_16_conv2d_14_kernel&
"assignvariableop_17_conv2d_14_bias7
3assignvariableop_18_instance_normalization_10_gamma6
2assignvariableop_19_instance_normalization_10_beta(
$assignvariableop_20_conv2d_15_kernel&
"assignvariableop_21_conv2d_15_bias7
3assignvariableop_22_instance_normalization_11_gamma6
2assignvariableop_23_instance_normalization_11_beta(
$assignvariableop_24_conv2d_16_kernel&
"assignvariableop_25_conv2d_16_bias7
3assignvariableop_26_instance_normalization_12_gamma6
2assignvariableop_27_instance_normalization_12_beta(
$assignvariableop_28_conv2d_17_kernel&
"assignvariableop_29_conv2d_17_bias
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp1assignvariableop_2_instance_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp0assignvariableop_3_instance_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_instance_normalization_7_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_instance_normalization_7_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp2assignvariableop_10_instance_normalization_8_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_instance_normalization_8_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_13_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_13_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp2assignvariableop_14_instance_normalization_9_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_instance_normalization_9_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_14_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_14_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp3assignvariableop_18_instance_normalization_10_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_instance_normalization_10_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_15_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_15_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_instance_normalization_11_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_instance_normalization_11_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_16_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_16_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_instance_normalization_12_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_instance_normalization_12_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_17_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_17_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30?
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*?
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
7__inference_instance_normalization_7_layer_call_fn_4617

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_25072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
q
G__inference_concatenate_1_layer_call_and_return_conditional_losses_2996

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????@??2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:?????????@??2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????@:?????????@?@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_2391

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addd
subSubinputsMean:output:0*
T0*1
_output_shapes
:??????????? 2
subk
truedivRealDivsub:z:0add:z:0*
T0*1
_output_shapes
:??????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapel
mulMultruediv:z:0Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1p
add_1AddV2mul:z:0Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?~
?
A__inference_model_3_layer_call_and_return_conditional_losses_3338

inputs
conv2d_10_3251
conv2d_10_3253!
instance_normalization_6_3257!
instance_normalization_6_3259
conv2d_11_3262
conv2d_11_3264!
instance_normalization_7_3268!
instance_normalization_7_3270
conv2d_12_3273
conv2d_12_3275!
instance_normalization_8_3279!
instance_normalization_8_3281
conv2d_13_3284
conv2d_13_3286!
instance_normalization_9_3290!
instance_normalization_9_3292
conv2d_14_3296
conv2d_14_3298"
instance_normalization_10_3301"
instance_normalization_10_3303
conv2d_15_3308
conv2d_15_3310"
instance_normalization_11_3313"
instance_normalization_11_3315
conv2d_16_3320
conv2d_16_3322"
instance_normalization_12_3325"
instance_normalization_12_3327
conv2d_17_3332
conv2d_17_3334
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?1instance_normalization_10/StatefulPartitionedCall?1instance_normalization_11/StatefulPartitionedCall?1instance_normalization_12/StatefulPartitionedCall?0instance_normalization_6/StatefulPartitionedCall?0instance_normalization_7/StatefulPartitionedCall?0instance_normalization_8/StatefulPartitionedCall?0instance_normalization_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_3251conv2d_10_3253*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_23092#
!conv2d_10/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_23302
leaky_re_lu_8/PartitionedCall?
0instance_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0instance_normalization_6_3257instance_normalization_6_3259*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_236422
0instance_normalization_6/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_6/StatefulPartitionedCall:output:0conv2d_11_3262conv2d_11_3264*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_24252#
!conv2d_11/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_24462
leaky_re_lu_9/PartitionedCall?
0instance_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0instance_normalization_7_3268instance_normalization_7_3270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_248022
0instance_normalization_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_7/StatefulPartitionedCall:output:0conv2d_12_3273conv2d_12_3275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_25412#
!conv2d_12/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_25622 
leaky_re_lu_10/PartitionedCall?
0instance_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0instance_normalization_8_3279instance_normalization_8_3281*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_259622
0instance_normalization_8/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_8/StatefulPartitionedCall:output:0conv2d_13_3284conv2d_13_3286*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_26572#
!conv2d_13/StatefulPartitionedCall?
leaky_re_lu_11/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_26782 
leaky_re_lu_11/PartitionedCall?
0instance_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0instance_normalization_9_3290instance_normalization_9_3292*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_271222
0instance_normalization_9/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall9instance_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_18482
up_sampling2d/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_14_3296conv2d_14_3298*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_27752#
!conv2d_14/StatefulPartitionedCall?
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0instance_normalization_10_3301instance_normalization_10_3303*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_281723
1instance_normalization_10/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:09instance_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_28752
concatenate/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_19952!
up_sampling2d_1/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_15_3308conv2d_15_3310*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_28962#
!conv2d_15/StatefulPartitionedCall?
1instance_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0instance_normalization_11_3313instance_normalization_11_3315*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_293823
1instance_normalization_11/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall:instance_normalization_11/StatefulPartitionedCall:output:09instance_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_29962
concatenate_1/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_21422!
up_sampling2d_2/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_16_3320conv2d_16_3322*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_30172#
!conv2d_16/StatefulPartitionedCall?
1instance_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0instance_normalization_12_3325instance_normalization_12_3327*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_305923
1instance_normalization_12/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall:instance_normalization_12/StatefulPartitionedCall:output:09instance_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_31172
concatenate_2/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_22892!
up_sampling2d_3/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_17_3332conv2d_17_3334*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_31382#
!conv2d_17/StatefulPartitionedCall?
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall2^instance_normalization_11/StatefulPartitionedCall2^instance_normalization_12/StatefulPartitionedCall1^instance_normalization_6/StatefulPartitionedCall1^instance_normalization_7/StatefulPartitionedCall1^instance_normalization_8/StatefulPartitionedCall1^instance_normalization_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2f
1instance_normalization_11/StatefulPartitionedCall1instance_normalization_11/StatefulPartitionedCall2f
1instance_normalization_12/StatefulPartitionedCall1instance_normalization_12/StatefulPartitionedCall2d
0instance_normalization_6/StatefulPartitionedCall0instance_normalization_6/StatefulPartitionedCall2d
0instance_normalization_7/StatefulPartitionedCall0instance_normalization_7/StatefulPartitionedCall2d
0instance_normalization_8/StatefulPartitionedCall0instance_normalization_8/StatefulPartitionedCall2d
0instance_normalization_9/StatefulPartitionedCall0instance_normalization_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
q
E__inference_concatenate_layer_call_and_return_conditional_losses_5134
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:????????? @?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,????????????????????????????:????????? @?:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:????????? @?
"
_user_specified_name
inputs/1
?~
?
A__inference_model_3_layer_call_and_return_conditional_losses_3155
input_3
conv2d_10_2320
conv2d_10_2322!
instance_normalization_6_2410!
instance_normalization_6_2412
conv2d_11_2436
conv2d_11_2438!
instance_normalization_7_2526!
instance_normalization_7_2528
conv2d_12_2552
conv2d_12_2554!
instance_normalization_8_2642!
instance_normalization_8_2644
conv2d_13_2668
conv2d_13_2670!
instance_normalization_9_2758!
instance_normalization_9_2760
conv2d_14_2786
conv2d_14_2788"
instance_normalization_10_2863"
instance_normalization_10_2865
conv2d_15_2907
conv2d_15_2909"
instance_normalization_11_2984"
instance_normalization_11_2986
conv2d_16_3028
conv2d_16_3030"
instance_normalization_12_3105"
instance_normalization_12_3107
conv2d_17_3149
conv2d_17_3151
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?1instance_normalization_10/StatefulPartitionedCall?1instance_normalization_11/StatefulPartitionedCall?1instance_normalization_12/StatefulPartitionedCall?0instance_normalization_6/StatefulPartitionedCall?0instance_normalization_7/StatefulPartitionedCall?0instance_normalization_8/StatefulPartitionedCall?0instance_normalization_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_10_2320conv2d_10_2322*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_23092#
!conv2d_10/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_23302
leaky_re_lu_8/PartitionedCall?
0instance_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0instance_normalization_6_2410instance_normalization_6_2412*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_236422
0instance_normalization_6/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_6/StatefulPartitionedCall:output:0conv2d_11_2436conv2d_11_2438*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_24252#
!conv2d_11/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_24462
leaky_re_lu_9/PartitionedCall?
0instance_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0instance_normalization_7_2526instance_normalization_7_2528*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_248022
0instance_normalization_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_7/StatefulPartitionedCall:output:0conv2d_12_2552conv2d_12_2554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_25412#
!conv2d_12/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_25622 
leaky_re_lu_10/PartitionedCall?
0instance_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0instance_normalization_8_2642instance_normalization_8_2644*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_259622
0instance_normalization_8/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_8/StatefulPartitionedCall:output:0conv2d_13_2668conv2d_13_2670*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_26572#
!conv2d_13/StatefulPartitionedCall?
leaky_re_lu_11/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_26782 
leaky_re_lu_11/PartitionedCall?
0instance_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0instance_normalization_9_2758instance_normalization_9_2760*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_271222
0instance_normalization_9/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall9instance_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_18482
up_sampling2d/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_14_2786conv2d_14_2788*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_27752#
!conv2d_14/StatefulPartitionedCall?
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0instance_normalization_10_2863instance_normalization_10_2865*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_281723
1instance_normalization_10/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:09instance_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_28752
concatenate/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_19952!
up_sampling2d_1/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_15_2907conv2d_15_2909*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_28962#
!conv2d_15/StatefulPartitionedCall?
1instance_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0instance_normalization_11_2984instance_normalization_11_2986*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_293823
1instance_normalization_11/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall:instance_normalization_11/StatefulPartitionedCall:output:09instance_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_29962
concatenate_1/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_21422!
up_sampling2d_2/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_16_3028conv2d_16_3030*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_30172#
!conv2d_16/StatefulPartitionedCall?
1instance_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0instance_normalization_12_3105instance_normalization_12_3107*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_305923
1instance_normalization_12/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall:instance_normalization_12/StatefulPartitionedCall:output:09instance_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_31172
concatenate_2/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_22892!
up_sampling2d_3/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_17_3149conv2d_17_3151*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_31382#
!conv2d_17/StatefulPartitionedCall?
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall2^instance_normalization_11/StatefulPartitionedCall2^instance_normalization_12/StatefulPartitionedCall1^instance_normalization_6/StatefulPartitionedCall1^instance_normalization_7/StatefulPartitionedCall1^instance_normalization_8/StatefulPartitionedCall1^instance_normalization_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2f
1instance_normalization_11/StatefulPartitionedCall1instance_normalization_11/StatefulPartitionedCall2f
1instance_normalization_12/StatefulPartitionedCall1instance_normalization_12/StatefulPartitionedCall2d
0instance_normalization_6/StatefulPartitionedCall0instance_normalization_6/StatefulPartitionedCall2d
0instance_normalization_7/StatefulPartitionedCall0instance_normalization_7/StatefulPartitionedCall2d
0instance_normalization_8/StatefulPartitionedCall0instance_normalization_8/StatefulPartitionedCall2d
0instance_normalization_9/StatefulPartitionedCall0instance_normalization_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
}
(__inference_conv2d_14_layer_call_fn_4983

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_27752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_2269

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_1700

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4599

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:?????????@?@2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:?????????@?@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_3059

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
A__inference_model_3_layer_call_and_return_conditional_losses_3882

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource<
8instance_normalization_6_reshape_readvariableop_resource>
:instance_normalization_6_reshape_1_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource<
8instance_normalization_7_reshape_readvariableop_resource>
:instance_normalization_7_reshape_1_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource<
8instance_normalization_8_reshape_readvariableop_resource>
:instance_normalization_8_reshape_1_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource<
8instance_normalization_9_reshape_readvariableop_resource>
:instance_normalization_9_reshape_1_readvariableop_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource=
9instance_normalization_10_reshape_readvariableop_resource?
;instance_normalization_10_reshape_1_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource=
9instance_normalization_11_reshape_readvariableop_resource?
;instance_normalization_11_reshape_1_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource=
9instance_normalization_12_reshape_readvariableop_resource?
;instance_normalization_12_reshape_1_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?0instance_normalization_10/Reshape/ReadVariableOp?2instance_normalization_10/Reshape_1/ReadVariableOp?0instance_normalization_11/Reshape/ReadVariableOp?2instance_normalization_11/Reshape_1/ReadVariableOp?0instance_normalization_12/Reshape/ReadVariableOp?2instance_normalization_12/Reshape_1/ReadVariableOp?/instance_normalization_6/Reshape/ReadVariableOp?1instance_normalization_6/Reshape_1/ReadVariableOp?/instance_normalization_7/Reshape/ReadVariableOp?1instance_normalization_7/Reshape_1/ReadVariableOp?/instance_normalization_8/Reshape/ReadVariableOp?1instance_normalization_8/Reshape_1/ReadVariableOp?/instance_normalization_9/Reshape/ReadVariableOp?1instance_normalization_9/Reshape_1/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_10/BiasAdd?
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_10/BiasAdd:output:0*1
_output_shapes
:??????????? 2
leaky_re_lu_8/LeakyRelu?
/instance_normalization_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_6/Mean/reduction_indices?
instance_normalization_6/MeanMean%leaky_re_lu_8/LeakyRelu:activations:08instance_normalization_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_6/Mean?
Jinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_6/reduce_std/reduce_variance/MeanMean%leaky_re_lu_8/LeakyRelu:activations:0Sinstance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_6/reduce_std/reduce_variance/Mean?
7instance_normalization_6/reduce_std/reduce_variance/subSub%leaky_re_lu_8/LeakyRelu:activations:0Ainstance_normalization_6/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 29
7instance_normalization_6/reduce_std/reduce_variance/sub?
:instance_normalization_6/reduce_std/reduce_variance/SquareSquare;instance_normalization_6/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2<
:instance_normalization_6/reduce_std/reduce_variance/Square?
Linstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_6/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_6/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_6/reduce_std/reduce_variance/Mean_1?
(instance_normalization_6/reduce_std/SqrtSqrtCinstance_normalization_6/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_6/reduce_std/Sqrt?
instance_normalization_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_6/add/y?
instance_normalization_6/addAddV2,instance_normalization_6/reduce_std/Sqrt:y:0'instance_normalization_6/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_6/add?
instance_normalization_6/subSub%leaky_re_lu_8/LeakyRelu:activations:0&instance_normalization_6/Mean:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_6/sub?
 instance_normalization_6/truedivRealDiv instance_normalization_6/sub:z:0 instance_normalization_6/add:z:0*
T0*1
_output_shapes
:??????????? 2"
 instance_normalization_6/truediv?
/instance_normalization_6/Reshape/ReadVariableOpReadVariableOp8instance_normalization_6_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_6/Reshape/ReadVariableOp?
&instance_normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_6/Reshape/shape?
 instance_normalization_6/ReshapeReshape7instance_normalization_6/Reshape/ReadVariableOp:value:0/instance_normalization_6/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_6/Reshape?
instance_normalization_6/mulMul$instance_normalization_6/truediv:z:0)instance_normalization_6/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_6/mul?
1instance_normalization_6/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_6_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_6/Reshape_1/ReadVariableOp?
(instance_normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_6/Reshape_1/shape?
"instance_normalization_6/Reshape_1Reshape9instance_normalization_6/Reshape_1/ReadVariableOp:value:01instance_normalization_6/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_6/Reshape_1?
instance_normalization_6/add_1AddV2 instance_normalization_6/mul:z:0+instance_normalization_6/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2 
instance_normalization_6/add_1?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D"instance_normalization_6/add_1:z:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_11/BiasAdd?
leaky_re_lu_9/LeakyRelu	LeakyReluconv2d_11/BiasAdd:output:0*0
_output_shapes
:?????????@?@2
leaky_re_lu_9/LeakyRelu?
/instance_normalization_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_7/Mean/reduction_indices?
instance_normalization_7/MeanMean%leaky_re_lu_9/LeakyRelu:activations:08instance_normalization_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_7/Mean?
Jinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_7/reduce_std/reduce_variance/MeanMean%leaky_re_lu_9/LeakyRelu:activations:0Sinstance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_7/reduce_std/reduce_variance/Mean?
7instance_normalization_7/reduce_std/reduce_variance/subSub%leaky_re_lu_9/LeakyRelu:activations:0Ainstance_normalization_7/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@29
7instance_normalization_7/reduce_std/reduce_variance/sub?
:instance_normalization_7/reduce_std/reduce_variance/SquareSquare;instance_normalization_7/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2<
:instance_normalization_7/reduce_std/reduce_variance/Square?
Linstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_7/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_7/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_7/reduce_std/reduce_variance/Mean_1?
(instance_normalization_7/reduce_std/SqrtSqrtCinstance_normalization_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_7/reduce_std/Sqrt?
instance_normalization_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_7/add/y?
instance_normalization_7/addAddV2,instance_normalization_7/reduce_std/Sqrt:y:0'instance_normalization_7/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_7/add?
instance_normalization_7/subSub%leaky_re_lu_9/LeakyRelu:activations:0&instance_normalization_7/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_7/sub?
 instance_normalization_7/truedivRealDiv instance_normalization_7/sub:z:0 instance_normalization_7/add:z:0*
T0*0
_output_shapes
:?????????@?@2"
 instance_normalization_7/truediv?
/instance_normalization_7/Reshape/ReadVariableOpReadVariableOp8instance_normalization_7_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_7/Reshape/ReadVariableOp?
&instance_normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_7/Reshape/shape?
 instance_normalization_7/ReshapeReshape7instance_normalization_7/Reshape/ReadVariableOp:value:0/instance_normalization_7/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_7/Reshape?
instance_normalization_7/mulMul$instance_normalization_7/truediv:z:0)instance_normalization_7/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_7/mul?
1instance_normalization_7/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_7_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_7/Reshape_1/ReadVariableOp?
(instance_normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_7/Reshape_1/shape?
"instance_normalization_7/Reshape_1Reshape9instance_normalization_7/Reshape_1/ReadVariableOp:value:01instance_normalization_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_7/Reshape_1?
instance_normalization_7/add_1AddV2 instance_normalization_7/mul:z:0+instance_normalization_7/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2 
instance_normalization_7/add_1?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D"instance_normalization_7/add_1:z:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
conv2d_12/BiasAdd?
leaky_re_lu_10/LeakyRelu	LeakyReluconv2d_12/BiasAdd:output:0*0
_output_shapes
:????????? @?2
leaky_re_lu_10/LeakyRelu?
/instance_normalization_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_8/Mean/reduction_indices?
instance_normalization_8/MeanMean&leaky_re_lu_10/LeakyRelu:activations:08instance_normalization_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_8/Mean?
Jinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_8/reduce_std/reduce_variance/MeanMean&leaky_re_lu_10/LeakyRelu:activations:0Sinstance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_8/reduce_std/reduce_variance/Mean?
7instance_normalization_8/reduce_std/reduce_variance/subSub&leaky_re_lu_10/LeakyRelu:activations:0Ainstance_normalization_8/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?29
7instance_normalization_8/reduce_std/reduce_variance/sub?
:instance_normalization_8/reduce_std/reduce_variance/SquareSquare;instance_normalization_8/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2<
:instance_normalization_8/reduce_std/reduce_variance/Square?
Linstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_8/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_8/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_8/reduce_std/reduce_variance/Mean_1?
(instance_normalization_8/reduce_std/SqrtSqrtCinstance_normalization_8/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_8/reduce_std/Sqrt?
instance_normalization_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_8/add/y?
instance_normalization_8/addAddV2,instance_normalization_8/reduce_std/Sqrt:y:0'instance_normalization_8/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_8/add?
instance_normalization_8/subSub&leaky_re_lu_10/LeakyRelu:activations:0&instance_normalization_8/Mean:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_8/sub?
 instance_normalization_8/truedivRealDiv instance_normalization_8/sub:z:0 instance_normalization_8/add:z:0*
T0*0
_output_shapes
:????????? @?2"
 instance_normalization_8/truediv?
/instance_normalization_8/Reshape/ReadVariableOpReadVariableOp8instance_normalization_8_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_8/Reshape/ReadVariableOp?
&instance_normalization_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_8/Reshape/shape?
 instance_normalization_8/ReshapeReshape7instance_normalization_8/Reshape/ReadVariableOp:value:0/instance_normalization_8/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_8/Reshape?
instance_normalization_8/mulMul$instance_normalization_8/truediv:z:0)instance_normalization_8/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_8/mul?
1instance_normalization_8/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_8_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_8/Reshape_1/ReadVariableOp?
(instance_normalization_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_8/Reshape_1/shape?
"instance_normalization_8/Reshape_1Reshape9instance_normalization_8/Reshape_1/ReadVariableOp:value:01instance_normalization_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_8/Reshape_1?
instance_normalization_8/add_1AddV2 instance_normalization_8/mul:z:0+instance_normalization_8/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2 
instance_normalization_8/add_1?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D"instance_normalization_8/add_1:z:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
conv2d_13/BiasAdd?
leaky_re_lu_11/LeakyRelu	LeakyReluconv2d_13/BiasAdd:output:0*0
_output_shapes
:????????? ?2
leaky_re_lu_11/LeakyRelu?
/instance_normalization_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/instance_normalization_9/Mean/reduction_indices?
instance_normalization_9/MeanMean&leaky_re_lu_11/LeakyRelu:activations:08instance_normalization_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
instance_normalization_9/Mean?
Jinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2L
Jinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices?
8instance_normalization_9/reduce_std/reduce_variance/MeanMean&leaky_re_lu_11/LeakyRelu:activations:0Sinstance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2:
8instance_normalization_9/reduce_std/reduce_variance/Mean?
7instance_normalization_9/reduce_std/reduce_variance/subSub&leaky_re_lu_11/LeakyRelu:activations:0Ainstance_normalization_9/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?29
7instance_normalization_9/reduce_std/reduce_variance/sub?
:instance_normalization_9/reduce_std/reduce_variance/SquareSquare;instance_normalization_9/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2<
:instance_normalization_9/reduce_std/reduce_variance/Square?
Linstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2N
Linstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices?
:instance_normalization_9/reduce_std/reduce_variance/Mean_1Mean>instance_normalization_9/reduce_std/reduce_variance/Square:y:0Uinstance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2<
:instance_normalization_9/reduce_std/reduce_variance/Mean_1?
(instance_normalization_9/reduce_std/SqrtSqrtCinstance_normalization_9/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2*
(instance_normalization_9/reduce_std/Sqrt?
instance_normalization_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2 
instance_normalization_9/add/y?
instance_normalization_9/addAddV2,instance_normalization_9/reduce_std/Sqrt:y:0'instance_normalization_9/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_9/add?
instance_normalization_9/subSub&leaky_re_lu_11/LeakyRelu:activations:0&instance_normalization_9/Mean:output:0*
T0*0
_output_shapes
:????????? ?2
instance_normalization_9/sub?
 instance_normalization_9/truedivRealDiv instance_normalization_9/sub:z:0 instance_normalization_9/add:z:0*
T0*0
_output_shapes
:????????? ?2"
 instance_normalization_9/truediv?
/instance_normalization_9/Reshape/ReadVariableOpReadVariableOp8instance_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/instance_normalization_9/Reshape/ReadVariableOp?
&instance_normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2(
&instance_normalization_9/Reshape/shape?
 instance_normalization_9/ReshapeReshape7instance_normalization_9/Reshape/ReadVariableOp:value:0/instance_normalization_9/Reshape/shape:output:0*
T0*&
_output_shapes
:2"
 instance_normalization_9/Reshape?
instance_normalization_9/mulMul$instance_normalization_9/truediv:z:0)instance_normalization_9/Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
instance_normalization_9/mul?
1instance_normalization_9/Reshape_1/ReadVariableOpReadVariableOp:instance_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1instance_normalization_9/Reshape_1/ReadVariableOp?
(instance_normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2*
(instance_normalization_9/Reshape_1/shape?
"instance_normalization_9/Reshape_1Reshape9instance_normalization_9/Reshape_1/ReadVariableOp:value:01instance_normalization_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2$
"instance_normalization_9/Reshape_1?
instance_normalization_9/add_1AddV2 instance_normalization_9/mul:z:0+instance_normalization_9/Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2 
instance_normalization_9/add_1|
up_sampling2d/ShapeShape"instance_normalization_9/add_1:z:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor"instance_normalization_9/add_1:z:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:????????? @?*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:????????? @?2
conv2d_14/Relu?
0instance_normalization_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_10/Mean/reduction_indices?
instance_normalization_10/MeanMeanconv2d_14/Relu:activations:09instance_normalization_10/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_10/Mean?
Kinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_10/reduce_std/reduce_variance/MeanMeanconv2d_14/Relu:activations:0Tinstance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_10/reduce_std/reduce_variance/Mean?
8instance_normalization_10/reduce_std/reduce_variance/subSubconv2d_14/Relu:activations:0Binstance_normalization_10/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2:
8instance_normalization_10/reduce_std/reduce_variance/sub?
;instance_normalization_10/reduce_std/reduce_variance/SquareSquare<instance_normalization_10/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2=
;instance_normalization_10/reduce_std/reduce_variance/Square?
Minstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_10/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_10/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_10/reduce_std/reduce_variance/Mean_1?
)instance_normalization_10/reduce_std/SqrtSqrtDinstance_normalization_10/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_10/reduce_std/Sqrt?
instance_normalization_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_10/add/y?
instance_normalization_10/addAddV2-instance_normalization_10/reduce_std/Sqrt:y:0(instance_normalization_10/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_10/add?
instance_normalization_10/subSubconv2d_14/Relu:activations:0'instance_normalization_10/Mean:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_10/sub?
!instance_normalization_10/truedivRealDiv!instance_normalization_10/sub:z:0!instance_normalization_10/add:z:0*
T0*0
_output_shapes
:????????? @?2#
!instance_normalization_10/truediv?
0instance_normalization_10/Reshape/ReadVariableOpReadVariableOp9instance_normalization_10_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_10/Reshape/ReadVariableOp?
'instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_10/Reshape/shape?
!instance_normalization_10/ReshapeReshape8instance_normalization_10/Reshape/ReadVariableOp:value:00instance_normalization_10/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_10/Reshape?
instance_normalization_10/mulMul%instance_normalization_10/truediv:z:0*instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
instance_normalization_10/mul?
2instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_10/Reshape_1/ReadVariableOp?
)instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_10/Reshape_1/shape?
#instance_normalization_10/Reshape_1Reshape:instance_normalization_10/Reshape_1/ReadVariableOp:value:02instance_normalization_10/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_10/Reshape_1?
instance_normalization_10/add_1AddV2!instance_normalization_10/mul:z:0,instance_normalization_10/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2!
instance_normalization_10/add_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2#instance_normalization_10/add_1:z:0"instance_normalization_8/add_1:z:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:????????? @?2
concatenate/concaty
up_sampling2d_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate/concat:output:0up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:?????????@??*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_15/BiasAdd
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@2
conv2d_15/Relu?
0instance_normalization_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_11/Mean/reduction_indices?
instance_normalization_11/MeanMeanconv2d_15/Relu:activations:09instance_normalization_11/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_11/Mean?
Kinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_11/reduce_std/reduce_variance/MeanMeanconv2d_15/Relu:activations:0Tinstance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_11/reduce_std/reduce_variance/Mean?
8instance_normalization_11/reduce_std/reduce_variance/subSubconv2d_15/Relu:activations:0Binstance_normalization_11/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2:
8instance_normalization_11/reduce_std/reduce_variance/sub?
;instance_normalization_11/reduce_std/reduce_variance/SquareSquare<instance_normalization_11/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2=
;instance_normalization_11/reduce_std/reduce_variance/Square?
Minstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_11/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_11/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_11/reduce_std/reduce_variance/Mean_1?
)instance_normalization_11/reduce_std/SqrtSqrtDinstance_normalization_11/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_11/reduce_std/Sqrt?
instance_normalization_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_11/add/y?
instance_normalization_11/addAddV2-instance_normalization_11/reduce_std/Sqrt:y:0(instance_normalization_11/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_11/add?
instance_normalization_11/subSubconv2d_15/Relu:activations:0'instance_normalization_11/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_11/sub?
!instance_normalization_11/truedivRealDiv!instance_normalization_11/sub:z:0!instance_normalization_11/add:z:0*
T0*0
_output_shapes
:?????????@?@2#
!instance_normalization_11/truediv?
0instance_normalization_11/Reshape/ReadVariableOpReadVariableOp9instance_normalization_11_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_11/Reshape/ReadVariableOp?
'instance_normalization_11/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_11/Reshape/shape?
!instance_normalization_11/ReshapeReshape8instance_normalization_11/Reshape/ReadVariableOp:value:00instance_normalization_11/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_11/Reshape?
instance_normalization_11/mulMul%instance_normalization_11/truediv:z:0*instance_normalization_11/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
instance_normalization_11/mul?
2instance_normalization_11/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_11_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_11/Reshape_1/ReadVariableOp?
)instance_normalization_11/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_11/Reshape_1/shape?
#instance_normalization_11/Reshape_1Reshape:instance_normalization_11/Reshape_1/ReadVariableOp:value:02instance_normalization_11/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_11/Reshape_1?
instance_normalization_11/add_1AddV2!instance_normalization_11/mul:z:0,instance_normalization_11/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2!
instance_normalization_11/add_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2#instance_normalization_11/add_1:z:0"instance_normalization_7/add_1:z:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????@??2
concatenate_1/concat{
up_sampling2d_2/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape?
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack?
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1?
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_1/concat:output:0up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/BiasAdd?
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_16/Relu?
0instance_normalization_12/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         22
0instance_normalization_12/Mean/reduction_indices?
instance_normalization_12/MeanMeanconv2d_16/Relu:activations:09instance_normalization_12/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2 
instance_normalization_12/Mean?
Kinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2M
Kinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices?
9instance_normalization_12/reduce_std/reduce_variance/MeanMeanconv2d_16/Relu:activations:0Tinstance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2;
9instance_normalization_12/reduce_std/reduce_variance/Mean?
8instance_normalization_12/reduce_std/reduce_variance/subSubconv2d_16/Relu:activations:0Binstance_normalization_12/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2:
8instance_normalization_12/reduce_std/reduce_variance/sub?
;instance_normalization_12/reduce_std/reduce_variance/SquareSquare<instance_normalization_12/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2=
;instance_normalization_12/reduce_std/reduce_variance/Square?
Minstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2O
Minstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices?
;instance_normalization_12/reduce_std/reduce_variance/Mean_1Mean?instance_normalization_12/reduce_std/reduce_variance/Square:y:0Vinstance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2=
;instance_normalization_12/reduce_std/reduce_variance/Mean_1?
)instance_normalization_12/reduce_std/SqrtSqrtDinstance_normalization_12/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2+
)instance_normalization_12/reduce_std/Sqrt?
instance_normalization_12/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2!
instance_normalization_12/add/y?
instance_normalization_12/addAddV2-instance_normalization_12/reduce_std/Sqrt:y:0(instance_normalization_12/add/y:output:0*
T0*/
_output_shapes
:?????????2
instance_normalization_12/add?
instance_normalization_12/subSubconv2d_16/Relu:activations:0'instance_normalization_12/Mean:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_12/sub?
!instance_normalization_12/truedivRealDiv!instance_normalization_12/sub:z:0!instance_normalization_12/add:z:0*
T0*1
_output_shapes
:??????????? 2#
!instance_normalization_12/truediv?
0instance_normalization_12/Reshape/ReadVariableOpReadVariableOp9instance_normalization_12_reshape_readvariableop_resource*
_output_shapes
:*
dtype022
0instance_normalization_12/Reshape/ReadVariableOp?
'instance_normalization_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2)
'instance_normalization_12/Reshape/shape?
!instance_normalization_12/ReshapeReshape8instance_normalization_12/Reshape/ReadVariableOp:value:00instance_normalization_12/Reshape/shape:output:0*
T0*&
_output_shapes
:2#
!instance_normalization_12/Reshape?
instance_normalization_12/mulMul%instance_normalization_12/truediv:z:0*instance_normalization_12/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
instance_normalization_12/mul?
2instance_normalization_12/Reshape_1/ReadVariableOpReadVariableOp;instance_normalization_12_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype024
2instance_normalization_12/Reshape_1/ReadVariableOp?
)instance_normalization_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)instance_normalization_12/Reshape_1/shape?
#instance_normalization_12/Reshape_1Reshape:instance_normalization_12/Reshape_1/ReadVariableOp:value:02instance_normalization_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2%
#instance_normalization_12/Reshape_1?
instance_normalization_12/add_1AddV2!instance_normalization_12/mul:z:0,instance_normalization_12/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2!
instance_normalization_12/add_1x
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2#instance_normalization_12/add_1:z:0"instance_normalization_6/add_1:z:0"concatenate_2/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatenate_2/concat{
up_sampling2d_3/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape?
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack?
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1?
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2?
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const?
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul?
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconcatenate_2/concat:output:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_17/BiasAdd?
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_17/Tanh?

IdentityIdentityconv2d_17/Tanh:y:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp1^instance_normalization_10/Reshape/ReadVariableOp3^instance_normalization_10/Reshape_1/ReadVariableOp1^instance_normalization_11/Reshape/ReadVariableOp3^instance_normalization_11/Reshape_1/ReadVariableOp1^instance_normalization_12/Reshape/ReadVariableOp3^instance_normalization_12/Reshape_1/ReadVariableOp0^instance_normalization_6/Reshape/ReadVariableOp2^instance_normalization_6/Reshape_1/ReadVariableOp0^instance_normalization_7/Reshape/ReadVariableOp2^instance_normalization_7/Reshape_1/ReadVariableOp0^instance_normalization_8/Reshape/ReadVariableOp2^instance_normalization_8/Reshape_1/ReadVariableOp0^instance_normalization_9/Reshape/ReadVariableOp2^instance_normalization_9/Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2d
0instance_normalization_10/Reshape/ReadVariableOp0instance_normalization_10/Reshape/ReadVariableOp2h
2instance_normalization_10/Reshape_1/ReadVariableOp2instance_normalization_10/Reshape_1/ReadVariableOp2d
0instance_normalization_11/Reshape/ReadVariableOp0instance_normalization_11/Reshape/ReadVariableOp2h
2instance_normalization_11/Reshape_1/ReadVariableOp2instance_normalization_11/Reshape_1/ReadVariableOp2d
0instance_normalization_12/Reshape/ReadVariableOp0instance_normalization_12/Reshape/ReadVariableOp2h
2instance_normalization_12/Reshape_1/ReadVariableOp2instance_normalization_12/Reshape_1/ReadVariableOp2b
/instance_normalization_6/Reshape/ReadVariableOp/instance_normalization_6/Reshape/ReadVariableOp2f
1instance_normalization_6/Reshape_1/ReadVariableOp1instance_normalization_6/Reshape_1/ReadVariableOp2b
/instance_normalization_7/Reshape/ReadVariableOp/instance_normalization_7/Reshape/ReadVariableOp2f
1instance_normalization_7/Reshape_1/ReadVariableOp1instance_normalization_7/Reshape_1/ReadVariableOp2b
/instance_normalization_8/Reshape/ReadVariableOp/instance_normalization_8/Reshape/ReadVariableOp2f
1instance_normalization_8/Reshape_1/ReadVariableOp1instance_normalization_8/Reshape_1/ReadVariableOp2b
/instance_normalization_9/Reshape/ReadVariableOp/instance_normalization_9/Reshape/ReadVariableOp2f
1instance_normalization_9/Reshape_1/ReadVariableOp1instance_normalization_9/Reshape_1/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5010

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_2938

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
o
E__inference_concatenate_layer_call_and_return_conditional_losses_2875

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:????????? @?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,????????????????????????????:????????? @?:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_2541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
H
,__inference_leaky_re_lu_8_layer_call_fn_4300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_23302
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_11_layer_call_fn_5295

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_29382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4745

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? @?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? @?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
s
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5311
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????@??2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:?????????@??2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????@:?????????@?@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????@?@
"
_user_specified_name
inputs/1
?
?
8__inference_instance_normalization_12_layer_call_fn_5409

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_30862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_12_layer_call_fn_5400

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_30592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_2844

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addu
subSubinputsMean:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
sub|
truedivRealDivsub:z:0add:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape}
mulMultruediv:z:0Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_2562

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:????????? @?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? @?:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_3086

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_2364

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addd
subSubinputsMean:output:0*
T0*1
_output_shapes
:??????????? 2
subk
truedivRealDivsub:z:0add:z:0*
T0*1
_output_shapes
:??????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapel
mulMultruediv:z:0Reshape:output:0*
T0*1
_output_shapes
:??????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1p
add_1AddV2mul:z:0Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_4814

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:????????? ?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? ?:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_2739

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? ?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? ?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_2309

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_2_layer_call_fn_2148

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_21422
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_12_layer_call_fn_5472

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_22332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4572

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:?????????@?@2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:?????????@?@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?D
?
__inference__traced_save_5627
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop=
9savev2_instance_normalization_6_gamma_read_readvariableop<
8savev2_instance_normalization_6_beta_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop=
9savev2_instance_normalization_7_gamma_read_readvariableop<
8savev2_instance_normalization_7_beta_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop=
9savev2_instance_normalization_8_gamma_read_readvariableop<
8savev2_instance_normalization_8_beta_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop=
9savev2_instance_normalization_9_gamma_read_readvariableop<
8savev2_instance_normalization_9_beta_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop>
:savev2_instance_normalization_10_gamma_read_readvariableop=
9savev2_instance_normalization_10_beta_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop>
:savev2_instance_normalization_11_gamma_read_readvariableop=
9savev2_instance_normalization_11_beta_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop>
:savev2_instance_normalization_12_gamma_read_readvariableop=
9savev2_instance_normalization_12_beta_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop9savev2_instance_normalization_6_gamma_read_readvariableop8savev2_instance_normalization_6_beta_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop9savev2_instance_normalization_7_gamma_read_readvariableop8savev2_instance_normalization_7_beta_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop9savev2_instance_normalization_8_gamma_read_readvariableop8savev2_instance_normalization_8_beta_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop9savev2_instance_normalization_9_gamma_read_readvariableop8savev2_instance_normalization_9_beta_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop:savev2_instance_normalization_10_gamma_read_readvariableop9savev2_instance_normalization_10_beta_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop:savev2_instance_normalization_11_gamma_read_readvariableop9savev2_instance_normalization_11_beta_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop:savev2_instance_normalization_12_gamma_read_readvariableop9savev2_instance_normalization_12_beta_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : ::: @:@:::@?:?:::??:?:::??:?:::?@:@:::? : :::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
::-	)
'
_output_shapes
:@?:!


_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?@: 

_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:? : 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
?
q
G__inference_concatenate_2_layer_call_and_return_conditional_losses_3117

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+??????????????????????????? :??????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:YU
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
V
*__inference_concatenate_layer_call_fn_5140
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_28752
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,????????????????????????????:????????? @?:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:????????? @?
"
_user_specified_name
inputs/1
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4772

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? @?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? @?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_2712

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? ?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? ?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_2507

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:?????????@?@2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:?????????@?@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_1939

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_13_layer_call_fn_4809

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_26572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4873

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? ?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? ?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? ?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_6_layer_call_fn_4435

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_14082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_1572

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1848

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_2596

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? @?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? @?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
c
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_4468

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@?@2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_6_layer_call_fn_4372

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_23912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_2122

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_8_layer_call_fn_4718

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_17002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_11_layer_call_fn_4463

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_24252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_2425

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
H
,__inference_leaky_re_lu_9_layer_call_fn_4473

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_24462
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_3_layer_call_fn_2295

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_22892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_leaky_re_lu_11_layer_call_fn_4819

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_26782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? ?:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1995

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_1536

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4426

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_2623

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addc
subSubinputsMean:output:0*
T0*0
_output_shapes
:????????? @?2
subj
truedivRealDivsub:z:0add:z:0*
T0*0
_output_shapes
:????????? @?2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshapek
mulMultruediv:z:0Reshape:output:0*
T0*0
_output_shapes
:????????? @?2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1o
add_1AddV2mul:z:0Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_7_layer_call_fn_4545

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_15722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_10_layer_call_fn_5046

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_19392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_17_layer_call_fn_5514

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_31382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4399

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_2330

inputs
identity^
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:??????????? 2
	LeakyReluu
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_2775

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_1792

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_2446

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@?@2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????@?@:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_11_layer_call_fn_5304

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_29652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_1444

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5463

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_7_layer_call_fn_4608

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_24802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@?@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_1664

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4673

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_9_layer_call_fn_4954

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_17922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_up_sampling2d_1_layer_call_fn_2001

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_19952
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_3_layer_call_fn_3401
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_33382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5286

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
&__inference_model_3_layer_call_fn_4271

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_34932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_4627

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????@?@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@?@
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4527

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?~
?
A__inference_model_3_layer_call_and_return_conditional_losses_3245
input_3
conv2d_10_3158
conv2d_10_3160!
instance_normalization_6_3164!
instance_normalization_6_3166
conv2d_11_3169
conv2d_11_3171!
instance_normalization_7_3175!
instance_normalization_7_3177
conv2d_12_3180
conv2d_12_3182!
instance_normalization_8_3186!
instance_normalization_8_3188
conv2d_13_3191
conv2d_13_3193!
instance_normalization_9_3197!
instance_normalization_9_3199
conv2d_14_3203
conv2d_14_3205"
instance_normalization_10_3208"
instance_normalization_10_3210
conv2d_15_3215
conv2d_15_3217"
instance_normalization_11_3220"
instance_normalization_11_3222
conv2d_16_3227
conv2d_16_3229"
instance_normalization_12_3232"
instance_normalization_12_3234
conv2d_17_3239
conv2d_17_3241
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?1instance_normalization_10/StatefulPartitionedCall?1instance_normalization_11/StatefulPartitionedCall?1instance_normalization_12/StatefulPartitionedCall?0instance_normalization_6/StatefulPartitionedCall?0instance_normalization_7/StatefulPartitionedCall?0instance_normalization_8/StatefulPartitionedCall?0instance_normalization_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_10_3158conv2d_10_3160*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_23092#
!conv2d_10/StatefulPartitionedCall?
leaky_re_lu_8/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_23302
leaky_re_lu_8/PartitionedCall?
0instance_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0instance_normalization_6_3164instance_normalization_6_3166*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_239122
0instance_normalization_6/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_6/StatefulPartitionedCall:output:0conv2d_11_3169conv2d_11_3171*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_24252#
!conv2d_11/StatefulPartitionedCall?
leaky_re_lu_9/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_24462
leaky_re_lu_9/PartitionedCall?
0instance_normalization_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0instance_normalization_7_3175instance_normalization_7_3177*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_250722
0instance_normalization_7/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_7/StatefulPartitionedCall:output:0conv2d_12_3180conv2d_12_3182*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_12_layer_call_and_return_conditional_losses_25412#
!conv2d_12/StatefulPartitionedCall?
leaky_re_lu_10/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_25622 
leaky_re_lu_10/PartitionedCall?
0instance_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0instance_normalization_8_3186instance_normalization_8_3188*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_262322
0instance_normalization_8/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall9instance_normalization_8/StatefulPartitionedCall:output:0conv2d_13_3191conv2d_13_3193*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_13_layer_call_and_return_conditional_losses_26572#
!conv2d_13/StatefulPartitionedCall?
leaky_re_lu_11/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_26782 
leaky_re_lu_11/PartitionedCall?
0instance_normalization_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0instance_normalization_9_3197instance_normalization_9_3199*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_273922
0instance_normalization_9/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall9instance_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_18482
up_sampling2d/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_14_3203conv2d_14_3205*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_27752#
!conv2d_14/StatefulPartitionedCall?
1instance_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0instance_normalization_10_3208instance_normalization_10_3210*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_284423
1instance_normalization_10/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall:instance_normalization_10/StatefulPartitionedCall:output:09instance_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_28752
concatenate/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_19952!
up_sampling2d_1/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_15_3215conv2d_15_3217*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_28962#
!conv2d_15/StatefulPartitionedCall?
1instance_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0instance_normalization_11_3220instance_normalization_11_3222*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_296523
1instance_normalization_11/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall:instance_normalization_11/StatefulPartitionedCall:output:09instance_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_29962
concatenate_1/PartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_21422!
up_sampling2d_2/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_16_3227conv2d_16_3229*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_30172#
!conv2d_16/StatefulPartitionedCall?
1instance_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0instance_normalization_12_3232instance_normalization_12_3234*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_308623
1instance_normalization_12/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall:instance_normalization_12/StatefulPartitionedCall:output:09instance_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_2_layer_call_and_return_conditional_losses_31172
concatenate_2/PartitionedCall?
up_sampling2d_3/PartitionedCallPartitionedCall&concatenate_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_22892!
up_sampling2d_3/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_17_3239conv2d_17_3241*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_31382#
!conv2d_17/StatefulPartitionedCall?
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall2^instance_normalization_10/StatefulPartitionedCall2^instance_normalization_11/StatefulPartitionedCall2^instance_normalization_12/StatefulPartitionedCall1^instance_normalization_6/StatefulPartitionedCall1^instance_normalization_7/StatefulPartitionedCall1^instance_normalization_8/StatefulPartitionedCall1^instance_normalization_9/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2f
1instance_normalization_10/StatefulPartitionedCall1instance_normalization_10/StatefulPartitionedCall2f
1instance_normalization_11/StatefulPartitionedCall1instance_normalization_11/StatefulPartitionedCall2f
1instance_normalization_12/StatefulPartitionedCall1instance_normalization_12/StatefulPartitionedCall2d
0instance_normalization_6/StatefulPartitionedCall0instance_normalization_6/StatefulPartitionedCall2d
0instance_normalization_7/StatefulPartitionedCall0instance_normalization_7/StatefulPartitionedCall2d
0instance_normalization_8/StatefulPartitionedCall0instance_normalization_8/StatefulPartitionedCall2d
0instance_normalization_9/StatefulPartitionedCall0instance_normalization_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?	
?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_2657

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? @?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_2233

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4700

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_10_layer_call_fn_5127

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_28442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_12_layer_call_fn_5481

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_22692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2142

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_17_layer_call_and_return_conditional_losses_3138

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_6_layer_call_fn_4444

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_14442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4945

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_11_layer_call_fn_5223

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_20862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_10_layer_call_fn_4290

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_23092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_3623
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_13232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
&__inference_model_3_layer_call_fn_3556
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_34932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
7__inference_instance_normalization_7_layer_call_fn_4536

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_15362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5391

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
addt
subSubinputsMean:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
sub{
truedivRealDivsub:z:0add:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape|
mulMultruediv:z:0Reshape:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
I
-__inference_leaky_re_lu_10_layer_call_fn_4646

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? @?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_25622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:????????? @?2

Identity"
identityIdentity:output:0*/
_input_shapes
:????????? @?:X T
0
_output_shapes
:????????? @?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_1408

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_9_layer_call_fn_4891

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_27392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
C__inference_conv2d_16_layer_call_and_return_conditional_losses_3017

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_9_layer_call_fn_4882

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_27122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:????????? ?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:????????? ?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_1828

inputs#
reshape_readvariableop_resource%
!reshape_1_readvariableop_resource
identity??Reshape/ReadVariableOp?Reshape_1/ReadVariableOp?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Mean/reduction_indices?
MeanMeaninputsMean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2
Mean?
1reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         23
1reduce_std/reduce_variance/Mean/reduction_indices?
reduce_std/reduce_variance/MeanMeaninputs:reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2!
reduce_std/reduce_variance/Mean?
reduce_std/reduce_variance/subSubinputs(reduce_std/reduce_variance/Mean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2 
reduce_std/reduce_variance/sub?
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2#
!reduce_std/reduce_variance/Square?
3reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         25
3reduce_std/reduce_variance/Mean_1/reduction_indices?
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0<reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2#
!reduce_std/reduce_variance/Mean_1?
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????2
reduce_std/SqrtS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
add/yr
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*/
_output_shapes
:?????????2
add}
subSubinputsMean:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
sub?
truedivRealDivsub:z:0add:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2	
truediv?
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape/shape?
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*&
_output_shapes
:2	
Reshape?
mulMultruediv:z:0Reshape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
mul?
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02
Reshape_1/ReadVariableOp{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape?
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1?
add_1AddV2mul:z:0Reshape_1:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
add_1?
IdentityIdentity	add_1:z:0^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_1323
input_34
0model_3_conv2d_10_conv2d_readvariableop_resource5
1model_3_conv2d_10_biasadd_readvariableop_resourceD
@model_3_instance_normalization_6_reshape_readvariableop_resourceF
Bmodel_3_instance_normalization_6_reshape_1_readvariableop_resource4
0model_3_conv2d_11_conv2d_readvariableop_resource5
1model_3_conv2d_11_biasadd_readvariableop_resourceD
@model_3_instance_normalization_7_reshape_readvariableop_resourceF
Bmodel_3_instance_normalization_7_reshape_1_readvariableop_resource4
0model_3_conv2d_12_conv2d_readvariableop_resource5
1model_3_conv2d_12_biasadd_readvariableop_resourceD
@model_3_instance_normalization_8_reshape_readvariableop_resourceF
Bmodel_3_instance_normalization_8_reshape_1_readvariableop_resource4
0model_3_conv2d_13_conv2d_readvariableop_resource5
1model_3_conv2d_13_biasadd_readvariableop_resourceD
@model_3_instance_normalization_9_reshape_readvariableop_resourceF
Bmodel_3_instance_normalization_9_reshape_1_readvariableop_resource4
0model_3_conv2d_14_conv2d_readvariableop_resource5
1model_3_conv2d_14_biasadd_readvariableop_resourceE
Amodel_3_instance_normalization_10_reshape_readvariableop_resourceG
Cmodel_3_instance_normalization_10_reshape_1_readvariableop_resource4
0model_3_conv2d_15_conv2d_readvariableop_resource5
1model_3_conv2d_15_biasadd_readvariableop_resourceE
Amodel_3_instance_normalization_11_reshape_readvariableop_resourceG
Cmodel_3_instance_normalization_11_reshape_1_readvariableop_resource4
0model_3_conv2d_16_conv2d_readvariableop_resource5
1model_3_conv2d_16_biasadd_readvariableop_resourceE
Amodel_3_instance_normalization_12_reshape_readvariableop_resourceG
Cmodel_3_instance_normalization_12_reshape_1_readvariableop_resource4
0model_3_conv2d_17_conv2d_readvariableop_resource5
1model_3_conv2d_17_biasadd_readvariableop_resource
identity??(model_3/conv2d_10/BiasAdd/ReadVariableOp?'model_3/conv2d_10/Conv2D/ReadVariableOp?(model_3/conv2d_11/BiasAdd/ReadVariableOp?'model_3/conv2d_11/Conv2D/ReadVariableOp?(model_3/conv2d_12/BiasAdd/ReadVariableOp?'model_3/conv2d_12/Conv2D/ReadVariableOp?(model_3/conv2d_13/BiasAdd/ReadVariableOp?'model_3/conv2d_13/Conv2D/ReadVariableOp?(model_3/conv2d_14/BiasAdd/ReadVariableOp?'model_3/conv2d_14/Conv2D/ReadVariableOp?(model_3/conv2d_15/BiasAdd/ReadVariableOp?'model_3/conv2d_15/Conv2D/ReadVariableOp?(model_3/conv2d_16/BiasAdd/ReadVariableOp?'model_3/conv2d_16/Conv2D/ReadVariableOp?(model_3/conv2d_17/BiasAdd/ReadVariableOp?'model_3/conv2d_17/Conv2D/ReadVariableOp?8model_3/instance_normalization_10/Reshape/ReadVariableOp?:model_3/instance_normalization_10/Reshape_1/ReadVariableOp?8model_3/instance_normalization_11/Reshape/ReadVariableOp?:model_3/instance_normalization_11/Reshape_1/ReadVariableOp?8model_3/instance_normalization_12/Reshape/ReadVariableOp?:model_3/instance_normalization_12/Reshape_1/ReadVariableOp?7model_3/instance_normalization_6/Reshape/ReadVariableOp?9model_3/instance_normalization_6/Reshape_1/ReadVariableOp?7model_3/instance_normalization_7/Reshape/ReadVariableOp?9model_3/instance_normalization_7/Reshape_1/ReadVariableOp?7model_3/instance_normalization_8/Reshape/ReadVariableOp?9model_3/instance_normalization_8/Reshape_1/ReadVariableOp?7model_3/instance_normalization_9/Reshape/ReadVariableOp?9model_3/instance_normalization_9/Reshape_1/ReadVariableOp?
'model_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_10/Conv2D/ReadVariableOp?
model_3/conv2d_10/Conv2DConv2Dinput_3/model_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_3/conv2d_10/Conv2D?
(model_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_10/BiasAdd/ReadVariableOp?
model_3/conv2d_10/BiasAddBiasAdd!model_3/conv2d_10/Conv2D:output:00model_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_10/BiasAdd?
model_3/leaky_re_lu_8/LeakyRelu	LeakyRelu"model_3/conv2d_10/BiasAdd:output:0*1
_output_shapes
:??????????? 2!
model_3/leaky_re_lu_8/LeakyRelu?
7model_3/instance_normalization_6/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         29
7model_3/instance_normalization_6/Mean/reduction_indices?
%model_3/instance_normalization_6/MeanMean-model_3/leaky_re_lu_8/LeakyRelu:activations:0@model_3/instance_normalization_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%model_3/instance_normalization_6/Mean?
Rmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2T
Rmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices?
@model_3/instance_normalization_6/reduce_std/reduce_variance/MeanMean-model_3/leaky_re_lu_8/LeakyRelu:activations:0[model_3/instance_normalization_6/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2B
@model_3/instance_normalization_6/reduce_std/reduce_variance/Mean?
?model_3/instance_normalization_6/reduce_std/reduce_variance/subSub-model_3/leaky_re_lu_8/LeakyRelu:activations:0Imodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2A
?model_3/instance_normalization_6/reduce_std/reduce_variance/sub?
Bmodel_3/instance_normalization_6/reduce_std/reduce_variance/SquareSquareCmodel_3/instance_normalization_6/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2D
Bmodel_3/instance_normalization_6/reduce_std/reduce_variance/Square?
Tmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2V
Tmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices?
Bmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1MeanFmodel_3/instance_normalization_6/reduce_std/reduce_variance/Square:y:0]model_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2D
Bmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1?
0model_3/instance_normalization_6/reduce_std/SqrtSqrtKmodel_3/instance_normalization_6/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????22
0model_3/instance_normalization_6/reduce_std/Sqrt?
&model_3/instance_normalization_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&model_3/instance_normalization_6/add/y?
$model_3/instance_normalization_6/addAddV24model_3/instance_normalization_6/reduce_std/Sqrt:y:0/model_3/instance_normalization_6/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$model_3/instance_normalization_6/add?
$model_3/instance_normalization_6/subSub-model_3/leaky_re_lu_8/LeakyRelu:activations:0.model_3/instance_normalization_6/Mean:output:0*
T0*1
_output_shapes
:??????????? 2&
$model_3/instance_normalization_6/sub?
(model_3/instance_normalization_6/truedivRealDiv(model_3/instance_normalization_6/sub:z:0(model_3/instance_normalization_6/add:z:0*
T0*1
_output_shapes
:??????????? 2*
(model_3/instance_normalization_6/truediv?
7model_3/instance_normalization_6/Reshape/ReadVariableOpReadVariableOp@model_3_instance_normalization_6_reshape_readvariableop_resource*
_output_shapes
:*
dtype029
7model_3/instance_normalization_6/Reshape/ReadVariableOp?
.model_3/instance_normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            20
.model_3/instance_normalization_6/Reshape/shape?
(model_3/instance_normalization_6/ReshapeReshape?model_3/instance_normalization_6/Reshape/ReadVariableOp:value:07model_3/instance_normalization_6/Reshape/shape:output:0*
T0*&
_output_shapes
:2*
(model_3/instance_normalization_6/Reshape?
$model_3/instance_normalization_6/mulMul,model_3/instance_normalization_6/truediv:z:01model_3/instance_normalization_6/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2&
$model_3/instance_normalization_6/mul?
9model_3/instance_normalization_6/Reshape_1/ReadVariableOpReadVariableOpBmodel_3_instance_normalization_6_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02;
9model_3/instance_normalization_6/Reshape_1/ReadVariableOp?
0model_3/instance_normalization_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model_3/instance_normalization_6/Reshape_1/shape?
*model_3/instance_normalization_6/Reshape_1ReshapeAmodel_3/instance_normalization_6/Reshape_1/ReadVariableOp:value:09model_3/instance_normalization_6/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2,
*model_3/instance_normalization_6/Reshape_1?
&model_3/instance_normalization_6/add_1AddV2(model_3/instance_normalization_6/mul:z:03model_3/instance_normalization_6/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2(
&model_3/instance_normalization_6/add_1?
'model_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_3/conv2d_11/Conv2D/ReadVariableOp?
model_3/conv2d_11/Conv2DConv2D*model_3/instance_normalization_6/add_1:z:0/model_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
model_3/conv2d_11/Conv2D?
(model_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_11/BiasAdd/ReadVariableOp?
model_3/conv2d_11/BiasAddBiasAdd!model_3/conv2d_11/Conv2D:output:00model_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
model_3/conv2d_11/BiasAdd?
model_3/leaky_re_lu_9/LeakyRelu	LeakyRelu"model_3/conv2d_11/BiasAdd:output:0*0
_output_shapes
:?????????@?@2!
model_3/leaky_re_lu_9/LeakyRelu?
7model_3/instance_normalization_7/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         29
7model_3/instance_normalization_7/Mean/reduction_indices?
%model_3/instance_normalization_7/MeanMean-model_3/leaky_re_lu_9/LeakyRelu:activations:0@model_3/instance_normalization_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%model_3/instance_normalization_7/Mean?
Rmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2T
Rmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices?
@model_3/instance_normalization_7/reduce_std/reduce_variance/MeanMean-model_3/leaky_re_lu_9/LeakyRelu:activations:0[model_3/instance_normalization_7/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2B
@model_3/instance_normalization_7/reduce_std/reduce_variance/Mean?
?model_3/instance_normalization_7/reduce_std/reduce_variance/subSub-model_3/leaky_re_lu_9/LeakyRelu:activations:0Imodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2A
?model_3/instance_normalization_7/reduce_std/reduce_variance/sub?
Bmodel_3/instance_normalization_7/reduce_std/reduce_variance/SquareSquareCmodel_3/instance_normalization_7/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2D
Bmodel_3/instance_normalization_7/reduce_std/reduce_variance/Square?
Tmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2V
Tmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices?
Bmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1MeanFmodel_3/instance_normalization_7/reduce_std/reduce_variance/Square:y:0]model_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2D
Bmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1?
0model_3/instance_normalization_7/reduce_std/SqrtSqrtKmodel_3/instance_normalization_7/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????22
0model_3/instance_normalization_7/reduce_std/Sqrt?
&model_3/instance_normalization_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&model_3/instance_normalization_7/add/y?
$model_3/instance_normalization_7/addAddV24model_3/instance_normalization_7/reduce_std/Sqrt:y:0/model_3/instance_normalization_7/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$model_3/instance_normalization_7/add?
$model_3/instance_normalization_7/subSub-model_3/leaky_re_lu_9/LeakyRelu:activations:0.model_3/instance_normalization_7/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2&
$model_3/instance_normalization_7/sub?
(model_3/instance_normalization_7/truedivRealDiv(model_3/instance_normalization_7/sub:z:0(model_3/instance_normalization_7/add:z:0*
T0*0
_output_shapes
:?????????@?@2*
(model_3/instance_normalization_7/truediv?
7model_3/instance_normalization_7/Reshape/ReadVariableOpReadVariableOp@model_3_instance_normalization_7_reshape_readvariableop_resource*
_output_shapes
:*
dtype029
7model_3/instance_normalization_7/Reshape/ReadVariableOp?
.model_3/instance_normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            20
.model_3/instance_normalization_7/Reshape/shape?
(model_3/instance_normalization_7/ReshapeReshape?model_3/instance_normalization_7/Reshape/ReadVariableOp:value:07model_3/instance_normalization_7/Reshape/shape:output:0*
T0*&
_output_shapes
:2*
(model_3/instance_normalization_7/Reshape?
$model_3/instance_normalization_7/mulMul,model_3/instance_normalization_7/truediv:z:01model_3/instance_normalization_7/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2&
$model_3/instance_normalization_7/mul?
9model_3/instance_normalization_7/Reshape_1/ReadVariableOpReadVariableOpBmodel_3_instance_normalization_7_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02;
9model_3/instance_normalization_7/Reshape_1/ReadVariableOp?
0model_3/instance_normalization_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model_3/instance_normalization_7/Reshape_1/shape?
*model_3/instance_normalization_7/Reshape_1ReshapeAmodel_3/instance_normalization_7/Reshape_1/ReadVariableOp:value:09model_3/instance_normalization_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2,
*model_3/instance_normalization_7/Reshape_1?
&model_3/instance_normalization_7/add_1AddV2(model_3/instance_normalization_7/mul:z:03model_3/instance_normalization_7/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2(
&model_3/instance_normalization_7/add_1?
'model_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'model_3/conv2d_12/Conv2D/ReadVariableOp?
model_3/conv2d_12/Conv2DConv2D*model_3/instance_normalization_7/add_1:z:0/model_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
model_3/conv2d_12/Conv2D?
(model_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_12/BiasAdd/ReadVariableOp?
model_3/conv2d_12/BiasAddBiasAdd!model_3/conv2d_12/Conv2D:output:00model_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
model_3/conv2d_12/BiasAdd?
 model_3/leaky_re_lu_10/LeakyRelu	LeakyRelu"model_3/conv2d_12/BiasAdd:output:0*0
_output_shapes
:????????? @?2"
 model_3/leaky_re_lu_10/LeakyRelu?
7model_3/instance_normalization_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         29
7model_3/instance_normalization_8/Mean/reduction_indices?
%model_3/instance_normalization_8/MeanMean.model_3/leaky_re_lu_10/LeakyRelu:activations:0@model_3/instance_normalization_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%model_3/instance_normalization_8/Mean?
Rmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2T
Rmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices?
@model_3/instance_normalization_8/reduce_std/reduce_variance/MeanMean.model_3/leaky_re_lu_10/LeakyRelu:activations:0[model_3/instance_normalization_8/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2B
@model_3/instance_normalization_8/reduce_std/reduce_variance/Mean?
?model_3/instance_normalization_8/reduce_std/reduce_variance/subSub.model_3/leaky_re_lu_10/LeakyRelu:activations:0Imodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2A
?model_3/instance_normalization_8/reduce_std/reduce_variance/sub?
Bmodel_3/instance_normalization_8/reduce_std/reduce_variance/SquareSquareCmodel_3/instance_normalization_8/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2D
Bmodel_3/instance_normalization_8/reduce_std/reduce_variance/Square?
Tmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2V
Tmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices?
Bmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1MeanFmodel_3/instance_normalization_8/reduce_std/reduce_variance/Square:y:0]model_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2D
Bmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1?
0model_3/instance_normalization_8/reduce_std/SqrtSqrtKmodel_3/instance_normalization_8/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????22
0model_3/instance_normalization_8/reduce_std/Sqrt?
&model_3/instance_normalization_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&model_3/instance_normalization_8/add/y?
$model_3/instance_normalization_8/addAddV24model_3/instance_normalization_8/reduce_std/Sqrt:y:0/model_3/instance_normalization_8/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$model_3/instance_normalization_8/add?
$model_3/instance_normalization_8/subSub.model_3/leaky_re_lu_10/LeakyRelu:activations:0.model_3/instance_normalization_8/Mean:output:0*
T0*0
_output_shapes
:????????? @?2&
$model_3/instance_normalization_8/sub?
(model_3/instance_normalization_8/truedivRealDiv(model_3/instance_normalization_8/sub:z:0(model_3/instance_normalization_8/add:z:0*
T0*0
_output_shapes
:????????? @?2*
(model_3/instance_normalization_8/truediv?
7model_3/instance_normalization_8/Reshape/ReadVariableOpReadVariableOp@model_3_instance_normalization_8_reshape_readvariableop_resource*
_output_shapes
:*
dtype029
7model_3/instance_normalization_8/Reshape/ReadVariableOp?
.model_3/instance_normalization_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            20
.model_3/instance_normalization_8/Reshape/shape?
(model_3/instance_normalization_8/ReshapeReshape?model_3/instance_normalization_8/Reshape/ReadVariableOp:value:07model_3/instance_normalization_8/Reshape/shape:output:0*
T0*&
_output_shapes
:2*
(model_3/instance_normalization_8/Reshape?
$model_3/instance_normalization_8/mulMul,model_3/instance_normalization_8/truediv:z:01model_3/instance_normalization_8/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2&
$model_3/instance_normalization_8/mul?
9model_3/instance_normalization_8/Reshape_1/ReadVariableOpReadVariableOpBmodel_3_instance_normalization_8_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02;
9model_3/instance_normalization_8/Reshape_1/ReadVariableOp?
0model_3/instance_normalization_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model_3/instance_normalization_8/Reshape_1/shape?
*model_3/instance_normalization_8/Reshape_1ReshapeAmodel_3/instance_normalization_8/Reshape_1/ReadVariableOp:value:09model_3/instance_normalization_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2,
*model_3/instance_normalization_8/Reshape_1?
&model_3/instance_normalization_8/add_1AddV2(model_3/instance_normalization_8/mul:z:03model_3/instance_normalization_8/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2(
&model_3/instance_normalization_8/add_1?
'model_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_3/conv2d_13/Conv2D/ReadVariableOp?
model_3/conv2d_13/Conv2DConv2D*model_3/instance_normalization_8/add_1:z:0/model_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
model_3/conv2d_13/Conv2D?
(model_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_13/BiasAdd/ReadVariableOp?
model_3/conv2d_13/BiasAddBiasAdd!model_3/conv2d_13/Conv2D:output:00model_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
model_3/conv2d_13/BiasAdd?
 model_3/leaky_re_lu_11/LeakyRelu	LeakyRelu"model_3/conv2d_13/BiasAdd:output:0*0
_output_shapes
:????????? ?2"
 model_3/leaky_re_lu_11/LeakyRelu?
7model_3/instance_normalization_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         29
7model_3/instance_normalization_9/Mean/reduction_indices?
%model_3/instance_normalization_9/MeanMean.model_3/leaky_re_lu_11/LeakyRelu:activations:0@model_3/instance_normalization_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2'
%model_3/instance_normalization_9/Mean?
Rmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2T
Rmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices?
@model_3/instance_normalization_9/reduce_std/reduce_variance/MeanMean.model_3/leaky_re_lu_11/LeakyRelu:activations:0[model_3/instance_normalization_9/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2B
@model_3/instance_normalization_9/reduce_std/reduce_variance/Mean?
?model_3/instance_normalization_9/reduce_std/reduce_variance/subSub.model_3/leaky_re_lu_11/LeakyRelu:activations:0Imodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? ?2A
?model_3/instance_normalization_9/reduce_std/reduce_variance/sub?
Bmodel_3/instance_normalization_9/reduce_std/reduce_variance/SquareSquareCmodel_3/instance_normalization_9/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? ?2D
Bmodel_3/instance_normalization_9/reduce_std/reduce_variance/Square?
Tmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2V
Tmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices?
Bmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1MeanFmodel_3/instance_normalization_9/reduce_std/reduce_variance/Square:y:0]model_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2D
Bmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1?
0model_3/instance_normalization_9/reduce_std/SqrtSqrtKmodel_3/instance_normalization_9/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????22
0model_3/instance_normalization_9/reduce_std/Sqrt?
&model_3/instance_normalization_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&model_3/instance_normalization_9/add/y?
$model_3/instance_normalization_9/addAddV24model_3/instance_normalization_9/reduce_std/Sqrt:y:0/model_3/instance_normalization_9/add/y:output:0*
T0*/
_output_shapes
:?????????2&
$model_3/instance_normalization_9/add?
$model_3/instance_normalization_9/subSub.model_3/leaky_re_lu_11/LeakyRelu:activations:0.model_3/instance_normalization_9/Mean:output:0*
T0*0
_output_shapes
:????????? ?2&
$model_3/instance_normalization_9/sub?
(model_3/instance_normalization_9/truedivRealDiv(model_3/instance_normalization_9/sub:z:0(model_3/instance_normalization_9/add:z:0*
T0*0
_output_shapes
:????????? ?2*
(model_3/instance_normalization_9/truediv?
7model_3/instance_normalization_9/Reshape/ReadVariableOpReadVariableOp@model_3_instance_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype029
7model_3/instance_normalization_9/Reshape/ReadVariableOp?
.model_3/instance_normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            20
.model_3/instance_normalization_9/Reshape/shape?
(model_3/instance_normalization_9/ReshapeReshape?model_3/instance_normalization_9/Reshape/ReadVariableOp:value:07model_3/instance_normalization_9/Reshape/shape:output:0*
T0*&
_output_shapes
:2*
(model_3/instance_normalization_9/Reshape?
$model_3/instance_normalization_9/mulMul,model_3/instance_normalization_9/truediv:z:01model_3/instance_normalization_9/Reshape:output:0*
T0*0
_output_shapes
:????????? ?2&
$model_3/instance_normalization_9/mul?
9model_3/instance_normalization_9/Reshape_1/ReadVariableOpReadVariableOpBmodel_3_instance_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02;
9model_3/instance_normalization_9/Reshape_1/ReadVariableOp?
0model_3/instance_normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            22
0model_3/instance_normalization_9/Reshape_1/shape?
*model_3/instance_normalization_9/Reshape_1ReshapeAmodel_3/instance_normalization_9/Reshape_1/ReadVariableOp:value:09model_3/instance_normalization_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2,
*model_3/instance_normalization_9/Reshape_1?
&model_3/instance_normalization_9/add_1AddV2(model_3/instance_normalization_9/mul:z:03model_3/instance_normalization_9/Reshape_1:output:0*
T0*0
_output_shapes
:????????? ?2(
&model_3/instance_normalization_9/add_1?
model_3/up_sampling2d/ShapeShape*model_3/instance_normalization_9/add_1:z:0*
T0*
_output_shapes
:2
model_3/up_sampling2d/Shape?
)model_3/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)model_3/up_sampling2d/strided_slice/stack?
+model_3/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_3/up_sampling2d/strided_slice/stack_1?
+model_3/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_3/up_sampling2d/strided_slice/stack_2?
#model_3/up_sampling2d/strided_sliceStridedSlice$model_3/up_sampling2d/Shape:output:02model_3/up_sampling2d/strided_slice/stack:output:04model_3/up_sampling2d/strided_slice/stack_1:output:04model_3/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#model_3/up_sampling2d/strided_slice?
model_3/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_3/up_sampling2d/Const?
model_3/up_sampling2d/mulMul,model_3/up_sampling2d/strided_slice:output:0$model_3/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d/mul?
2model_3/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor*model_3/instance_normalization_9/add_1:z:0model_3/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:????????? @?*
half_pixel_centers(24
2model_3/up_sampling2d/resize/ResizeNearestNeighbor?
'model_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_3/conv2d_14/Conv2D/ReadVariableOp?
model_3/conv2d_14/Conv2DConv2DCmodel_3/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0/model_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?*
paddingSAME*
strides
2
model_3/conv2d_14/Conv2D?
(model_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_3/conv2d_14/BiasAdd/ReadVariableOp?
model_3/conv2d_14/BiasAddBiasAdd!model_3/conv2d_14/Conv2D:output:00model_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? @?2
model_3/conv2d_14/BiasAdd?
model_3/conv2d_14/ReluRelu"model_3/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:????????? @?2
model_3/conv2d_14/Relu?
8model_3/instance_normalization_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2:
8model_3/instance_normalization_10/Mean/reduction_indices?
&model_3/instance_normalization_10/MeanMean$model_3/conv2d_14/Relu:activations:0Amodel_3/instance_normalization_10/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2(
&model_3/instance_normalization_10/Mean?
Smodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2U
Smodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices?
Amodel_3/instance_normalization_10/reduce_std/reduce_variance/MeanMean$model_3/conv2d_14/Relu:activations:0\model_3/instance_normalization_10/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2C
Amodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean?
@model_3/instance_normalization_10/reduce_std/reduce_variance/subSub$model_3/conv2d_14/Relu:activations:0Jmodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:????????? @?2B
@model_3/instance_normalization_10/reduce_std/reduce_variance/sub?
Cmodel_3/instance_normalization_10/reduce_std/reduce_variance/SquareSquareDmodel_3/instance_normalization_10/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:????????? @?2E
Cmodel_3/instance_normalization_10/reduce_std/reduce_variance/Square?
Umodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Umodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices?
Cmodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1MeanGmodel_3/instance_normalization_10/reduce_std/reduce_variance/Square:y:0^model_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2E
Cmodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1?
1model_3/instance_normalization_10/reduce_std/SqrtSqrtLmodel_3/instance_normalization_10/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????23
1model_3/instance_normalization_10/reduce_std/Sqrt?
'model_3/instance_normalization_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2)
'model_3/instance_normalization_10/add/y?
%model_3/instance_normalization_10/addAddV25model_3/instance_normalization_10/reduce_std/Sqrt:y:00model_3/instance_normalization_10/add/y:output:0*
T0*/
_output_shapes
:?????????2'
%model_3/instance_normalization_10/add?
%model_3/instance_normalization_10/subSub$model_3/conv2d_14/Relu:activations:0/model_3/instance_normalization_10/Mean:output:0*
T0*0
_output_shapes
:????????? @?2'
%model_3/instance_normalization_10/sub?
)model_3/instance_normalization_10/truedivRealDiv)model_3/instance_normalization_10/sub:z:0)model_3/instance_normalization_10/add:z:0*
T0*0
_output_shapes
:????????? @?2+
)model_3/instance_normalization_10/truediv?
8model_3/instance_normalization_10/Reshape/ReadVariableOpReadVariableOpAmodel_3_instance_normalization_10_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_3/instance_normalization_10/Reshape/ReadVariableOp?
/model_3/instance_normalization_10/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model_3/instance_normalization_10/Reshape/shape?
)model_3/instance_normalization_10/ReshapeReshape@model_3/instance_normalization_10/Reshape/ReadVariableOp:value:08model_3/instance_normalization_10/Reshape/shape:output:0*
T0*&
_output_shapes
:2+
)model_3/instance_normalization_10/Reshape?
%model_3/instance_normalization_10/mulMul-model_3/instance_normalization_10/truediv:z:02model_3/instance_normalization_10/Reshape:output:0*
T0*0
_output_shapes
:????????? @?2'
%model_3/instance_normalization_10/mul?
:model_3/instance_normalization_10/Reshape_1/ReadVariableOpReadVariableOpCmodel_3_instance_normalization_10_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_3/instance_normalization_10/Reshape_1/ReadVariableOp?
1model_3/instance_normalization_10/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1model_3/instance_normalization_10/Reshape_1/shape?
+model_3/instance_normalization_10/Reshape_1ReshapeBmodel_3/instance_normalization_10/Reshape_1/ReadVariableOp:value:0:model_3/instance_normalization_10/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2-
+model_3/instance_normalization_10/Reshape_1?
'model_3/instance_normalization_10/add_1AddV2)model_3/instance_normalization_10/mul:z:04model_3/instance_normalization_10/Reshape_1:output:0*
T0*0
_output_shapes
:????????? @?2)
'model_3/instance_normalization_10/add_1?
model_3/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model_3/concatenate/concat/axis?
model_3/concatenate/concatConcatV2+model_3/instance_normalization_10/add_1:z:0*model_3/instance_normalization_8/add_1:z:0(model_3/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:????????? @?2
model_3/concatenate/concat?
model_3/up_sampling2d_1/ShapeShape#model_3/concatenate/concat:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_1/Shape?
+model_3/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_3/up_sampling2d_1/strided_slice/stack?
-model_3/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_1/strided_slice/stack_1?
-model_3/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_1/strided_slice/stack_2?
%model_3/up_sampling2d_1/strided_sliceStridedSlice&model_3/up_sampling2d_1/Shape:output:04model_3/up_sampling2d_1/strided_slice/stack:output:06model_3/up_sampling2d_1/strided_slice/stack_1:output:06model_3/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_3/up_sampling2d_1/strided_slice?
model_3/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_3/up_sampling2d_1/Const?
model_3/up_sampling2d_1/mulMul.model_3/up_sampling2d_1/strided_slice:output:0&model_3/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_1/mul?
4model_3/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#model_3/concatenate/concat:output:0model_3/up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:?????????@??*
half_pixel_centers(26
4model_3/up_sampling2d_1/resize/ResizeNearestNeighbor?
'model_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'model_3/conv2d_15/Conv2D/ReadVariableOp?
model_3/conv2d_15/Conv2DConv2DEmodel_3/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0/model_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@*
paddingSAME*
strides
2
model_3/conv2d_15/Conv2D?
(model_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_15/BiasAdd/ReadVariableOp?
model_3/conv2d_15/BiasAddBiasAdd!model_3/conv2d_15/Conv2D:output:00model_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?@2
model_3/conv2d_15/BiasAdd?
model_3/conv2d_15/ReluRelu"model_3/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?@2
model_3/conv2d_15/Relu?
8model_3/instance_normalization_11/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2:
8model_3/instance_normalization_11/Mean/reduction_indices?
&model_3/instance_normalization_11/MeanMean$model_3/conv2d_15/Relu:activations:0Amodel_3/instance_normalization_11/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2(
&model_3/instance_normalization_11/Mean?
Smodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2U
Smodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices?
Amodel_3/instance_normalization_11/reduce_std/reduce_variance/MeanMean$model_3/conv2d_15/Relu:activations:0\model_3/instance_normalization_11/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2C
Amodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean?
@model_3/instance_normalization_11/reduce_std/reduce_variance/subSub$model_3/conv2d_15/Relu:activations:0Jmodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2B
@model_3/instance_normalization_11/reduce_std/reduce_variance/sub?
Cmodel_3/instance_normalization_11/reduce_std/reduce_variance/SquareSquareDmodel_3/instance_normalization_11/reduce_std/reduce_variance/sub:z:0*
T0*0
_output_shapes
:?????????@?@2E
Cmodel_3/instance_normalization_11/reduce_std/reduce_variance/Square?
Umodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Umodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices?
Cmodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1MeanGmodel_3/instance_normalization_11/reduce_std/reduce_variance/Square:y:0^model_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2E
Cmodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1?
1model_3/instance_normalization_11/reduce_std/SqrtSqrtLmodel_3/instance_normalization_11/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????23
1model_3/instance_normalization_11/reduce_std/Sqrt?
'model_3/instance_normalization_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2)
'model_3/instance_normalization_11/add/y?
%model_3/instance_normalization_11/addAddV25model_3/instance_normalization_11/reduce_std/Sqrt:y:00model_3/instance_normalization_11/add/y:output:0*
T0*/
_output_shapes
:?????????2'
%model_3/instance_normalization_11/add?
%model_3/instance_normalization_11/subSub$model_3/conv2d_15/Relu:activations:0/model_3/instance_normalization_11/Mean:output:0*
T0*0
_output_shapes
:?????????@?@2'
%model_3/instance_normalization_11/sub?
)model_3/instance_normalization_11/truedivRealDiv)model_3/instance_normalization_11/sub:z:0)model_3/instance_normalization_11/add:z:0*
T0*0
_output_shapes
:?????????@?@2+
)model_3/instance_normalization_11/truediv?
8model_3/instance_normalization_11/Reshape/ReadVariableOpReadVariableOpAmodel_3_instance_normalization_11_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_3/instance_normalization_11/Reshape/ReadVariableOp?
/model_3/instance_normalization_11/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model_3/instance_normalization_11/Reshape/shape?
)model_3/instance_normalization_11/ReshapeReshape@model_3/instance_normalization_11/Reshape/ReadVariableOp:value:08model_3/instance_normalization_11/Reshape/shape:output:0*
T0*&
_output_shapes
:2+
)model_3/instance_normalization_11/Reshape?
%model_3/instance_normalization_11/mulMul-model_3/instance_normalization_11/truediv:z:02model_3/instance_normalization_11/Reshape:output:0*
T0*0
_output_shapes
:?????????@?@2'
%model_3/instance_normalization_11/mul?
:model_3/instance_normalization_11/Reshape_1/ReadVariableOpReadVariableOpCmodel_3_instance_normalization_11_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_3/instance_normalization_11/Reshape_1/ReadVariableOp?
1model_3/instance_normalization_11/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1model_3/instance_normalization_11/Reshape_1/shape?
+model_3/instance_normalization_11/Reshape_1ReshapeBmodel_3/instance_normalization_11/Reshape_1/ReadVariableOp:value:0:model_3/instance_normalization_11/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2-
+model_3/instance_normalization_11/Reshape_1?
'model_3/instance_normalization_11/add_1AddV2)model_3/instance_normalization_11/mul:z:04model_3/instance_normalization_11/Reshape_1:output:0*
T0*0
_output_shapes
:?????????@?@2)
'model_3/instance_normalization_11/add_1?
!model_3/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_1/concat/axis?
model_3/concatenate_1/concatConcatV2+model_3/instance_normalization_11/add_1:z:0*model_3/instance_normalization_7/add_1:z:0*model_3/concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????@??2
model_3/concatenate_1/concat?
model_3/up_sampling2d_2/ShapeShape%model_3/concatenate_1/concat:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_2/Shape?
+model_3/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_3/up_sampling2d_2/strided_slice/stack?
-model_3/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_2/strided_slice/stack_1?
-model_3/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_2/strided_slice/stack_2?
%model_3/up_sampling2d_2/strided_sliceStridedSlice&model_3/up_sampling2d_2/Shape:output:04model_3/up_sampling2d_2/strided_slice/stack:output:06model_3/up_sampling2d_2/strided_slice/stack_1:output:06model_3/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_3/up_sampling2d_2/strided_slice?
model_3/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_3/up_sampling2d_2/Const?
model_3/up_sampling2d_2/mulMul.model_3/up_sampling2d_2/strided_slice:output:0&model_3/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_2/mul?
4model_3/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%model_3/concatenate_1/concat:output:0model_3/up_sampling2d_2/mul:z:0*
T0*2
_output_shapes 
:????????????*
half_pixel_centers(26
4model_3/up_sampling2d_2/resize/ResizeNearestNeighbor?
'model_3/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype02)
'model_3/conv2d_16/Conv2D/ReadVariableOp?
model_3/conv2d_16/Conv2DConv2DEmodel_3/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0/model_3/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model_3/conv2d_16/Conv2D?
(model_3/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_16/BiasAdd/ReadVariableOp?
model_3/conv2d_16/BiasAddBiasAdd!model_3/conv2d_16/Conv2D:output:00model_3/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_16/BiasAdd?
model_3/conv2d_16/ReluRelu"model_3/conv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model_3/conv2d_16/Relu?
8model_3/instance_normalization_12/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2:
8model_3/instance_normalization_12/Mean/reduction_indices?
&model_3/instance_normalization_12/MeanMean$model_3/conv2d_16/Relu:activations:0Amodel_3/instance_normalization_12/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2(
&model_3/instance_normalization_12/Mean?
Smodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2U
Smodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices?
Amodel_3/instance_normalization_12/reduce_std/reduce_variance/MeanMean$model_3/conv2d_16/Relu:activations:0\model_3/instance_normalization_12/reduce_std/reduce_variance/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2C
Amodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean?
@model_3/instance_normalization_12/reduce_std/reduce_variance/subSub$model_3/conv2d_16/Relu:activations:0Jmodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean:output:0*
T0*1
_output_shapes
:??????????? 2B
@model_3/instance_normalization_12/reduce_std/reduce_variance/sub?
Cmodel_3/instance_normalization_12/reduce_std/reduce_variance/SquareSquareDmodel_3/instance_normalization_12/reduce_std/reduce_variance/sub:z:0*
T0*1
_output_shapes
:??????????? 2E
Cmodel_3/instance_normalization_12/reduce_std/reduce_variance/Square?
Umodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Umodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices?
Cmodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1MeanGmodel_3/instance_normalization_12/reduce_std/reduce_variance/Square:y:0^model_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(2E
Cmodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1?
1model_3/instance_normalization_12/reduce_std/SqrtSqrtLmodel_3/instance_normalization_12/reduce_std/reduce_variance/Mean_1:output:0*
T0*/
_output_shapes
:?????????23
1model_3/instance_normalization_12/reduce_std/Sqrt?
'model_3/instance_normalization_12/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2)
'model_3/instance_normalization_12/add/y?
%model_3/instance_normalization_12/addAddV25model_3/instance_normalization_12/reduce_std/Sqrt:y:00model_3/instance_normalization_12/add/y:output:0*
T0*/
_output_shapes
:?????????2'
%model_3/instance_normalization_12/add?
%model_3/instance_normalization_12/subSub$model_3/conv2d_16/Relu:activations:0/model_3/instance_normalization_12/Mean:output:0*
T0*1
_output_shapes
:??????????? 2'
%model_3/instance_normalization_12/sub?
)model_3/instance_normalization_12/truedivRealDiv)model_3/instance_normalization_12/sub:z:0)model_3/instance_normalization_12/add:z:0*
T0*1
_output_shapes
:??????????? 2+
)model_3/instance_normalization_12/truediv?
8model_3/instance_normalization_12/Reshape/ReadVariableOpReadVariableOpAmodel_3_instance_normalization_12_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_3/instance_normalization_12/Reshape/ReadVariableOp?
/model_3/instance_normalization_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/model_3/instance_normalization_12/Reshape/shape?
)model_3/instance_normalization_12/ReshapeReshape@model_3/instance_normalization_12/Reshape/ReadVariableOp:value:08model_3/instance_normalization_12/Reshape/shape:output:0*
T0*&
_output_shapes
:2+
)model_3/instance_normalization_12/Reshape?
%model_3/instance_normalization_12/mulMul-model_3/instance_normalization_12/truediv:z:02model_3/instance_normalization_12/Reshape:output:0*
T0*1
_output_shapes
:??????????? 2'
%model_3/instance_normalization_12/mul?
:model_3/instance_normalization_12/Reshape_1/ReadVariableOpReadVariableOpCmodel_3_instance_normalization_12_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_3/instance_normalization_12/Reshape_1/ReadVariableOp?
1model_3/instance_normalization_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1model_3/instance_normalization_12/Reshape_1/shape?
+model_3/instance_normalization_12/Reshape_1ReshapeBmodel_3/instance_normalization_12/Reshape_1/ReadVariableOp:value:0:model_3/instance_normalization_12/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2-
+model_3/instance_normalization_12/Reshape_1?
'model_3/instance_normalization_12/add_1AddV2)model_3/instance_normalization_12/mul:z:04model_3/instance_normalization_12/Reshape_1:output:0*
T0*1
_output_shapes
:??????????? 2)
'model_3/instance_normalization_12/add_1?
!model_3/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_2/concat/axis?
model_3/concatenate_2/concatConcatV2+model_3/instance_normalization_12/add_1:z:0*model_3/instance_normalization_6/add_1:z:0*model_3/concatenate_2/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@2
model_3/concatenate_2/concat?
model_3/up_sampling2d_3/ShapeShape%model_3/concatenate_2/concat:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_3/Shape?
+model_3/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_3/up_sampling2d_3/strided_slice/stack?
-model_3/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_3/strided_slice/stack_1?
-model_3/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_3/up_sampling2d_3/strided_slice/stack_2?
%model_3/up_sampling2d_3/strided_sliceStridedSlice&model_3/up_sampling2d_3/Shape:output:04model_3/up_sampling2d_3/strided_slice/stack:output:06model_3/up_sampling2d_3/strided_slice/stack_1:output:06model_3/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_3/up_sampling2d_3/strided_slice?
model_3/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_3/up_sampling2d_3/Const?
model_3/up_sampling2d_3/mulMul.model_3/up_sampling2d_3/strided_slice:output:0&model_3/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_3/mul?
4model_3/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%model_3/concatenate_2/concat:output:0model_3/up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:???????????@*
half_pixel_centers(26
4model_3/up_sampling2d_3/resize/ResizeNearestNeighbor?
'model_3/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'model_3/conv2d_17/Conv2D/ReadVariableOp?
model_3/conv2d_17/Conv2DConv2DEmodel_3/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0/model_3/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_3/conv2d_17/Conv2D?
(model_3/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_17/BiasAdd/ReadVariableOp?
model_3/conv2d_17/BiasAddBiasAdd!model_3/conv2d_17/Conv2D:output:00model_3/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_17/BiasAdd?
model_3/conv2d_17/TanhTanh"model_3/conv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_17/Tanh?
IdentityIdentitymodel_3/conv2d_17/Tanh:y:0)^model_3/conv2d_10/BiasAdd/ReadVariableOp(^model_3/conv2d_10/Conv2D/ReadVariableOp)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp)^model_3/conv2d_14/BiasAdd/ReadVariableOp(^model_3/conv2d_14/Conv2D/ReadVariableOp)^model_3/conv2d_15/BiasAdd/ReadVariableOp(^model_3/conv2d_15/Conv2D/ReadVariableOp)^model_3/conv2d_16/BiasAdd/ReadVariableOp(^model_3/conv2d_16/Conv2D/ReadVariableOp)^model_3/conv2d_17/BiasAdd/ReadVariableOp(^model_3/conv2d_17/Conv2D/ReadVariableOp9^model_3/instance_normalization_10/Reshape/ReadVariableOp;^model_3/instance_normalization_10/Reshape_1/ReadVariableOp9^model_3/instance_normalization_11/Reshape/ReadVariableOp;^model_3/instance_normalization_11/Reshape_1/ReadVariableOp9^model_3/instance_normalization_12/Reshape/ReadVariableOp;^model_3/instance_normalization_12/Reshape_1/ReadVariableOp8^model_3/instance_normalization_6/Reshape/ReadVariableOp:^model_3/instance_normalization_6/Reshape_1/ReadVariableOp8^model_3/instance_normalization_7/Reshape/ReadVariableOp:^model_3/instance_normalization_7/Reshape_1/ReadVariableOp8^model_3/instance_normalization_8/Reshape/ReadVariableOp:^model_3/instance_normalization_8/Reshape_1/ReadVariableOp8^model_3/instance_normalization_9/Reshape/ReadVariableOp:^model_3/instance_normalization_9/Reshape_1/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::2T
(model_3/conv2d_10/BiasAdd/ReadVariableOp(model_3/conv2d_10/BiasAdd/ReadVariableOp2R
'model_3/conv2d_10/Conv2D/ReadVariableOp'model_3/conv2d_10/Conv2D/ReadVariableOp2T
(model_3/conv2d_11/BiasAdd/ReadVariableOp(model_3/conv2d_11/BiasAdd/ReadVariableOp2R
'model_3/conv2d_11/Conv2D/ReadVariableOp'model_3/conv2d_11/Conv2D/ReadVariableOp2T
(model_3/conv2d_12/BiasAdd/ReadVariableOp(model_3/conv2d_12/BiasAdd/ReadVariableOp2R
'model_3/conv2d_12/Conv2D/ReadVariableOp'model_3/conv2d_12/Conv2D/ReadVariableOp2T
(model_3/conv2d_13/BiasAdd/ReadVariableOp(model_3/conv2d_13/BiasAdd/ReadVariableOp2R
'model_3/conv2d_13/Conv2D/ReadVariableOp'model_3/conv2d_13/Conv2D/ReadVariableOp2T
(model_3/conv2d_14/BiasAdd/ReadVariableOp(model_3/conv2d_14/BiasAdd/ReadVariableOp2R
'model_3/conv2d_14/Conv2D/ReadVariableOp'model_3/conv2d_14/Conv2D/ReadVariableOp2T
(model_3/conv2d_15/BiasAdd/ReadVariableOp(model_3/conv2d_15/BiasAdd/ReadVariableOp2R
'model_3/conv2d_15/Conv2D/ReadVariableOp'model_3/conv2d_15/Conv2D/ReadVariableOp2T
(model_3/conv2d_16/BiasAdd/ReadVariableOp(model_3/conv2d_16/BiasAdd/ReadVariableOp2R
'model_3/conv2d_16/Conv2D/ReadVariableOp'model_3/conv2d_16/Conv2D/ReadVariableOp2T
(model_3/conv2d_17/BiasAdd/ReadVariableOp(model_3/conv2d_17/BiasAdd/ReadVariableOp2R
'model_3/conv2d_17/Conv2D/ReadVariableOp'model_3/conv2d_17/Conv2D/ReadVariableOp2t
8model_3/instance_normalization_10/Reshape/ReadVariableOp8model_3/instance_normalization_10/Reshape/ReadVariableOp2x
:model_3/instance_normalization_10/Reshape_1/ReadVariableOp:model_3/instance_normalization_10/Reshape_1/ReadVariableOp2t
8model_3/instance_normalization_11/Reshape/ReadVariableOp8model_3/instance_normalization_11/Reshape/ReadVariableOp2x
:model_3/instance_normalization_11/Reshape_1/ReadVariableOp:model_3/instance_normalization_11/Reshape_1/ReadVariableOp2t
8model_3/instance_normalization_12/Reshape/ReadVariableOp8model_3/instance_normalization_12/Reshape/ReadVariableOp2x
:model_3/instance_normalization_12/Reshape_1/ReadVariableOp:model_3/instance_normalization_12/Reshape_1/ReadVariableOp2r
7model_3/instance_normalization_6/Reshape/ReadVariableOp7model_3/instance_normalization_6/Reshape/ReadVariableOp2v
9model_3/instance_normalization_6/Reshape_1/ReadVariableOp9model_3/instance_normalization_6/Reshape_1/ReadVariableOp2r
7model_3/instance_normalization_7/Reshape/ReadVariableOp7model_3/instance_normalization_7/Reshape/ReadVariableOp2v
9model_3/instance_normalization_7/Reshape_1/ReadVariableOp9model_3/instance_normalization_7/Reshape_1/ReadVariableOp2r
7model_3/instance_normalization_8/Reshape/ReadVariableOp7model_3/instance_normalization_8/Reshape/ReadVariableOp2v
9model_3/instance_normalization_8/Reshape_1/ReadVariableOp9model_3/instance_normalization_8/Reshape_1/ReadVariableOp2r
7model_3/instance_normalization_9/Reshape/ReadVariableOp7model_3/instance_normalization_9/Reshape/ReadVariableOp2v
9model_3/instance_normalization_9/Reshape_1/ReadVariableOp9model_3/instance_normalization_9/Reshape_1/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?	
?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_4281

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_instance_normalization_10_layer_call_fn_5055

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_19752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_instance_normalization_9_layer_call_fn_4963

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_18282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:4????????????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_3:
serving_default_input_3:0???????????G
	conv2d_17:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer-20
layer-21
layer_with_weights-12
layer-22
layer_with_weights-13
layer-23
layer-24
layer-25
layer_with_weights-14
layer-26
trainable_variables
regularization_losses
	variables
	keras_api
 
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_6", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_6", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["instance_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_7", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_7", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["instance_normalization_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_8", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_8", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["instance_normalization_8", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_9", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_9", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["instance_normalization_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_10", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["instance_normalization_10", 0, 0, {}], ["instance_normalization_8", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_11", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_11", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["instance_normalization_11", 0, 0, {}], ["instance_normalization_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_12", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_12", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["instance_normalization_12", 0, 0, {}], ["instance_normalization_6", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["conv2d_17", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256, 512, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 512, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_6", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_6", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["instance_normalization_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_7", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_7", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["instance_normalization_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_8", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_8", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["instance_normalization_8", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_9", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_9", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["instance_normalization_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_10", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["instance_normalization_10", 0, 0, {}], ["instance_normalization_8", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_11", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_11", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["instance_normalization_11", 0, 0, {}], ["instance_normalization_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "InstanceNormalization", "config": {"name": "instance_normalization_12", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "instance_normalization_12", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["instance_normalization_12", 0, 0, {}], ["instance_normalization_6", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["conv2d_17", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 512, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?	

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 512, 1]}}
?
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
	+gamma
,beta
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_6", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 256, 32]}}
?	

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 256, 32]}}
?
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
	;gamma
<beta
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_7", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 64]}}
?	

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 64]}}
?
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
	Kgamma
Lbeta
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_8", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 64, 128]}}
?	

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 64, 128]}}
?
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
	[gamma
\beta
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_9", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 32, 256]}}
?
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

ekernel
fbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 64, 256]}}
?
	kgamma
lbeta
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_10", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 64, 128]}}
?
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 64, 128]}, {"class_name": "TensorShape", "items": [null, 32, 64, 128]}]}
?
utrainable_variables
vregularization_losses
w	variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

ykernel
zbias
{trainable_variables
|regularization_losses
}	variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 256]}}
?
	gamma
	?beta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_11", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 64]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 128, 64]}, {"class_name": "TensorShape", "items": [null, 64, 128, 64]}]}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 256, 128]}}
?

?gamma
	?beta
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InstanceNormalization", "name": "instance_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "instance_normalization_12", "trainable": true, "dtype": "float32", "axis": null, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 256, 32]}}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 256, 32]}, {"class_name": "TensorShape", "items": [null, 128, 256, 32]}]}
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
?kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 512, 64]}}
?
!0
"1
+2
,3
14
25
;6
<7
A8
B9
K10
L11
Q12
R13
[14
\15
e16
f17
k18
l19
y20
z21
22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!0
"1
+2
,3
14
25
;6
<7
A8
B9
K10
L11
Q12
R13
[14
\15
e16
f17
k18
l19
y20
z21
22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
trainable_variables
?layer_metrics
regularization_losses
?layers
?metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_10/kernel
: 2conv2d_10/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
#trainable_variables
?layer_metrics
$regularization_losses
?layers
?metrics
%	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
'trainable_variables
?layer_metrics
(regularization_losses
?layers
?metrics
)	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_6/gamma
+:)2instance_normalization_6/beta
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
-trainable_variables
?layer_metrics
.regularization_losses
?layers
?metrics
/	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_11/kernel
:@2conv2d_11/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
3trainable_variables
?layer_metrics
4regularization_losses
?layers
?metrics
5	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
7trainable_variables
?layer_metrics
8regularization_losses
?layers
?metrics
9	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_7/gamma
+:)2instance_normalization_7/beta
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
=trainable_variables
?layer_metrics
>regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_12/kernel
:?2conv2d_12/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Ctrainable_variables
?layer_metrics
Dregularization_losses
?layers
?metrics
E	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Gtrainable_variables
?layer_metrics
Hregularization_losses
?layers
?metrics
I	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_8/gamma
+:)2instance_normalization_8/beta
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Mtrainable_variables
?layer_metrics
Nregularization_losses
?layers
?metrics
O	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_13/kernel
:?2conv2d_13/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Strainable_variables
?layer_metrics
Tregularization_losses
?layers
?metrics
U	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Wtrainable_variables
?layer_metrics
Xregularization_losses
?layers
?metrics
Y	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*2instance_normalization_9/gamma
+:)2instance_normalization_9/beta
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
]trainable_variables
?layer_metrics
^regularization_losses
?layers
?metrics
_	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
atrainable_variables
?layer_metrics
bregularization_losses
?layers
?metrics
c	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_14/kernel
:?2conv2d_14/bias
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
gtrainable_variables
?layer_metrics
hregularization_losses
?layers
?metrics
i	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+2instance_normalization_10/gamma
,:*2instance_normalization_10/beta
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
mtrainable_variables
?layer_metrics
nregularization_losses
?layers
?metrics
o	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
qtrainable_variables
?layer_metrics
rregularization_losses
?layers
?metrics
s	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
utrainable_variables
?layer_metrics
vregularization_losses
?layers
?metrics
w	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_15/kernel
:@2conv2d_15/bias
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
{trainable_variables
?layer_metrics
|regularization_losses
?layers
?metrics
}	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+2instance_normalization_11/gamma
,:*2instance_normalization_11/beta
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_16/kernel
: 2conv2d_16/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+2instance_normalization_12/gamma
,:*2instance_normalization_12/beta
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_17/kernel
:2conv2d_17/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?layers
?metrics
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
A__inference_model_3_layer_call_and_return_conditional_losses_3155
A__inference_model_3_layer_call_and_return_conditional_losses_3882
A__inference_model_3_layer_call_and_return_conditional_losses_3245
A__inference_model_3_layer_call_and_return_conditional_losses_4141?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_1323?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_3???????????
?2?
&__inference_model_3_layer_call_fn_3401
&__inference_model_3_layer_call_fn_4271
&__inference_model_3_layer_call_fn_3556
&__inference_model_3_layer_call_fn_4206?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_10_layer_call_and_return_conditional_losses_4281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_10_layer_call_fn_4290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_4295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_leaky_re_lu_8_layer_call_fn_4300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4426
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4354
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4399
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4327?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_instance_normalization_6_layer_call_fn_4444
7__inference_instance_normalization_6_layer_call_fn_4435
7__inference_instance_normalization_6_layer_call_fn_4363
7__inference_instance_normalization_6_layer_call_fn_4372?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_4454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_11_layer_call_fn_4463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_4468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_leaky_re_lu_9_layer_call_fn_4473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4500
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4527
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4572
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4599?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_instance_normalization_7_layer_call_fn_4617
7__inference_instance_normalization_7_layer_call_fn_4536
7__inference_instance_normalization_7_layer_call_fn_4608
7__inference_instance_normalization_7_layer_call_fn_4545?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_4627?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_12_layer_call_fn_4636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_4641?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leaky_re_lu_10_layer_call_fn_4646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4673
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4745
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4772
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4700?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_instance_normalization_8_layer_call_fn_4790
7__inference_instance_normalization_8_layer_call_fn_4709
7__inference_instance_normalization_8_layer_call_fn_4718
7__inference_instance_normalization_8_layer_call_fn_4781?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_13_layer_call_and_return_conditional_losses_4800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_13_layer_call_fn_4809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_4814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leaky_re_lu_11_layer_call_fn_4819?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4846
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4918
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4945
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4873?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_instance_normalization_9_layer_call_fn_4954
7__inference_instance_normalization_9_layer_call_fn_4882
7__inference_instance_normalization_9_layer_call_fn_4891
7__inference_instance_normalization_9_layer_call_fn_4963?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_up_sampling2d_layer_call_fn_1854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_4974?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_14_layer_call_fn_4983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5082
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5010
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5037
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5109?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_instance_normalization_10_layer_call_fn_5127
8__inference_instance_normalization_10_layer_call_fn_5055
8__inference_instance_normalization_10_layer_call_fn_5046
8__inference_instance_normalization_10_layer_call_fn_5118?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_concatenate_layer_call_and_return_conditional_losses_5134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_concatenate_layer_call_fn_5140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_up_sampling2d_1_layer_call_fn_2001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_5151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_15_layer_call_fn_5160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5214
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5187
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5286
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5259?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_instance_normalization_11_layer_call_fn_5223
8__inference_instance_normalization_11_layer_call_fn_5304
8__inference_instance_normalization_11_layer_call_fn_5295
8__inference_instance_normalization_11_layer_call_fn_5232?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_1_layer_call_fn_5317?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_up_sampling2d_2_layer_call_fn_2148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_16_layer_call_fn_5337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5463
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5436
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5391
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5364?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_instance_normalization_12_layer_call_fn_5481
8__inference_instance_normalization_12_layer_call_fn_5472
8__inference_instance_normalization_12_layer_call_fn_5409
8__inference_instance_normalization_12_layer_call_fn_5400?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_2_layer_call_fn_5494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_up_sampling2d_3_layer_call_fn_2295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_17_layer_call_fn_5514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_3623input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_1323?%!"+,12;<ABKLQR[\efklyz???????:?7
0?-
+?(
input_3???????????
? "??<
:
	conv2d_17-?*
	conv2d_17????????????
G__inference_concatenate_1_layer_call_and_return_conditional_losses_5311?}?z
s?p
n?k
<?9
inputs/0+???????????????????????????@
+?(
inputs/1?????????@?@
? "/?,
%?"
0?????????@??
? ?
,__inference_concatenate_1_layer_call_fn_5317?}?z
s?p
n?k
<?9
inputs/0+???????????????????????????@
+?(
inputs/1?????????@?@
? ""??????????@???
G__inference_concatenate_2_layer_call_and_return_conditional_losses_5488?~?{
t?q
o?l
<?9
inputs/0+??????????????????????????? 
,?)
inputs/1??????????? 
? "/?,
%?"
0???????????@
? ?
,__inference_concatenate_2_layer_call_fn_5494?~?{
t?q
o?l
<?9
inputs/0+??????????????????????????? 
,?)
inputs/1??????????? 
? ""????????????@?
E__inference_concatenate_layer_call_and_return_conditional_losses_5134?~?{
t?q
o?l
=?:
inputs/0,????????????????????????????
+?(
inputs/1????????? @?
? ".?+
$?!
0????????? @?
? ?
*__inference_concatenate_layer_call_fn_5140?~?{
t?q
o?l
=?:
inputs/0,????????????????????????????
+?(
inputs/1????????? @?
? "!?????????? @??
C__inference_conv2d_10_layer_call_and_return_conditional_losses_4281p!"9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
(__inference_conv2d_10_layer_call_fn_4290c!"9?6
/?,
*?'
inputs???????????
? ""???????????? ?
C__inference_conv2d_11_layer_call_and_return_conditional_losses_4454o129?6
/?,
*?'
inputs??????????? 
? ".?+
$?!
0?????????@?@
? ?
(__inference_conv2d_11_layer_call_fn_4463b129?6
/?,
*?'
inputs??????????? 
? "!??????????@?@?
C__inference_conv2d_12_layer_call_and_return_conditional_losses_4627nAB8?5
.?+
)?&
inputs?????????@?@
? ".?+
$?!
0????????? @?
? ?
(__inference_conv2d_12_layer_call_fn_4636aAB8?5
.?+
)?&
inputs?????????@?@
? "!?????????? @??
C__inference_conv2d_13_layer_call_and_return_conditional_losses_4800nQR8?5
.?+
)?&
inputs????????? @?
? ".?+
$?!
0????????? ?
? ?
(__inference_conv2d_13_layer_call_fn_4809aQR8?5
.?+
)?&
inputs????????? @?
? "!?????????? ??
C__inference_conv2d_14_layer_call_and_return_conditional_losses_4974?efJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_conv2d_14_layer_call_fn_4983?efJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_conv2d_15_layer_call_and_return_conditional_losses_5151?yzJ?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_15_layer_call_fn_5160?yzJ?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5328???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
(__inference_conv2d_16_layer_call_fn_5337???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+??????????????????????????? ?
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5505???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
(__inference_conv2d_17_layer_call_fn_5514???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5010?klV?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5037?klV?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5082?klN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_instance_normalization_10_layer_call_and_return_conditional_losses_5109?klN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_instance_normalization_10_layer_call_fn_5046?klV?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
8__inference_instance_normalization_10_layer_call_fn_5055?klV?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
8__inference_instance_normalization_10_layer_call_fn_5118?klN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
8__inference_instance_normalization_10_layer_call_fn_5127?klN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5187??V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5214??V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5259??M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_instance_normalization_11_layer_call_and_return_conditional_losses_5286??M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_instance_normalization_11_layer_call_fn_5223??V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
8__inference_instance_normalization_11_layer_call_fn_5232??V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
8__inference_instance_normalization_11_layer_call_fn_5295??M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_instance_normalization_11_layer_call_fn_5304??M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5364???M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5391???M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5436???V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
S__inference_instance_normalization_12_layer_call_and_return_conditional_losses_5463???V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
8__inference_instance_normalization_12_layer_call_fn_5400???M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
8__inference_instance_normalization_12_layer_call_fn_5409???M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_instance_normalization_12_layer_call_fn_5472???V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
8__inference_instance_normalization_12_layer_call_fn_5481???V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4327t+,=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4354t+,=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4399?+,V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_6_layer_call_and_return_conditional_losses_4426?+,V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_instance_normalization_6_layer_call_fn_4363g+,=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
7__inference_instance_normalization_6_layer_call_fn_4372g+,=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
7__inference_instance_normalization_6_layer_call_fn_4435?+,V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
7__inference_instance_normalization_6_layer_call_fn_4444?+,V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4500?;<V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4527?;<V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4572r;<<?9
2?/
)?&
inputs?????????@?@
p
? ".?+
$?!
0?????????@?@
? ?
R__inference_instance_normalization_7_layer_call_and_return_conditional_losses_4599r;<<?9
2?/
)?&
inputs?????????@?@
p 
? ".?+
$?!
0?????????@?@
? ?
7__inference_instance_normalization_7_layer_call_fn_4536?;<V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
7__inference_instance_normalization_7_layer_call_fn_4545?;<V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
7__inference_instance_normalization_7_layer_call_fn_4608e;<<?9
2?/
)?&
inputs?????????@?@
p
? "!??????????@?@?
7__inference_instance_normalization_7_layer_call_fn_4617e;<<?9
2?/
)?&
inputs?????????@?@
p 
? "!??????????@?@?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4673?KLV?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4700?KLV?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4745rKL<?9
2?/
)?&
inputs????????? @?
p
? ".?+
$?!
0????????? @?
? ?
R__inference_instance_normalization_8_layer_call_and_return_conditional_losses_4772rKL<?9
2?/
)?&
inputs????????? @?
p 
? ".?+
$?!
0????????? @?
? ?
7__inference_instance_normalization_8_layer_call_fn_4709?KLV?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
7__inference_instance_normalization_8_layer_call_fn_4718?KLV?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
7__inference_instance_normalization_8_layer_call_fn_4781eKL<?9
2?/
)?&
inputs????????? @?
p
? "!?????????? @??
7__inference_instance_normalization_8_layer_call_fn_4790eKL<?9
2?/
)?&
inputs????????? @?
p 
? "!?????????? @??
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4846r[\<?9
2?/
)?&
inputs????????? ?
p
? ".?+
$?!
0????????? ?
? ?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4873r[\<?9
2?/
)?&
inputs????????? ?
p 
? ".?+
$?!
0????????? ?
? ?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4918?[\V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
R__inference_instance_normalization_9_layer_call_and_return_conditional_losses_4945?[\V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_instance_normalization_9_layer_call_fn_4882e[\<?9
2?/
)?&
inputs????????? ?
p
? "!?????????? ??
7__inference_instance_normalization_9_layer_call_fn_4891e[\<?9
2?/
)?&
inputs????????? ?
p 
? "!?????????? ??
7__inference_instance_normalization_9_layer_call_fn_4954?[\V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
7__inference_instance_normalization_9_layer_call_fn_4963?[\V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
H__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_4641j8?5
.?+
)?&
inputs????????? @?
? ".?+
$?!
0????????? @?
? ?
-__inference_leaky_re_lu_10_layer_call_fn_4646]8?5
.?+
)?&
inputs????????? @?
? "!?????????? @??
H__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_4814j8?5
.?+
)?&
inputs????????? ?
? ".?+
$?!
0????????? ?
? ?
-__inference_leaky_re_lu_11_layer_call_fn_4819]8?5
.?+
)?&
inputs????????? ?
? "!?????????? ??
G__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_4295l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
,__inference_leaky_re_lu_8_layer_call_fn_4300_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
G__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_4468j8?5
.?+
)?&
inputs?????????@?@
? ".?+
$?!
0?????????@?@
? ?
,__inference_leaky_re_lu_9_layer_call_fn_4473]8?5
.?+
)?&
inputs?????????@?@
? "!??????????@?@?
A__inference_model_3_layer_call_and_return_conditional_losses_3155?%!"+,12;<ABKLQR[\efklyz???????B??
8?5
+?(
input_3???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_3_layer_call_and_return_conditional_losses_3245?%!"+,12;<ABKLQR[\efklyz???????B??
8?5
+?(
input_3???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_3_layer_call_and_return_conditional_losses_3882?%!"+,12;<ABKLQR[\efklyz???????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
A__inference_model_3_layer_call_and_return_conditional_losses_4141?%!"+,12;<ABKLQR[\efklyz???????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
&__inference_model_3_layer_call_fn_3401?%!"+,12;<ABKLQR[\efklyz???????B??
8?5
+?(
input_3???????????
p

 
? "2?/+????????????????????????????
&__inference_model_3_layer_call_fn_3556?%!"+,12;<ABKLQR[\efklyz???????B??
8?5
+?(
input_3???????????
p 

 
? "2?/+????????????????????????????
&__inference_model_3_layer_call_fn_4206?%!"+,12;<ABKLQR[\efklyz???????A?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
&__inference_model_3_layer_call_fn_4271?%!"+,12;<ABKLQR[\efklyz???????A?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
"__inference_signature_wrapper_3623?%!"+,12;<ABKLQR[\efklyz???????E?B
? 
;?8
6
input_3+?(
input_3???????????"??<
:
	conv2d_17-?*
	conv2d_17????????????
I__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1995?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_1_layer_call_fn_2001?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_2142?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_2_layer_call_fn_2148?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_2289?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_up_sampling2d_3_layer_call_fn_2295?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_1848?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_up_sampling2d_layer_call_fn_1854?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????