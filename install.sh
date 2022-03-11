
# =========================
# DEFINE
# =========================



# input
CONDA_ENV='nvnmd'
NVNMD_PATH="nvnmd"
PHASE_ST=0
SUBPH_ST=0
PHASE_ED=100
SUBPH_ED=100

# base
INSTALL_PATH=`pwd`
PACKAGE="$INSTALL_PATH/package"
PATCH="$INSTALL_PATH/patch"
LOG_FN="$INSTALL_PATH/install.log"

# conda
CONDA_VERSION="4.11.0"
ENV_PATH=""

# gcc
GCC_VERSION="9.0.0"

# git
GIT_VERSION="2.34.0"

# python
PYTHON_VERSION="3.9.0"

# cmake
CMAKE_VERSION="3.10.0"
CMAKE="cmake"
CMAKE_BIN=$INSTALL_PATH/cmake/$CMAKE/bin/

# tensorflow
TENSORFLOW_VERSION="2.7.0"
TENSORFLOW="tensorflow"
LIBTF_CC_VERSION=2.7.0
LIBTF_CC_BUILD=cpu_h6ddf1b9_0

# nvnmd
NVNMD_VERSION="2.0.0"
NVNMD="nvnmd"
NVNMD_BIN="$INSTALL_PATH/nvnmd/bin/"

# head
HEAD_INFO="\033[32m #INFO \033[0m"
HEAD_WARNING="\033[33m #WARNING \033[0m"
HEAD_ERROR="\033[31m #ERROR \033[0m"

# flga
FLAG_RIGHT="\033[32m [v] \033[0m"
FLAG_NRIGHT="\033[33m [ ] \033[0m"
FLAG_FALSE="\033[31m [x] \033[0m"

# funtion return
RT_MATCH_VERSION=0
RT_FIND_CMD=0
RT_FIND_ENV=0

# global
phase=0
ist_phase=0
nxt_phase=0

subph=0
ist_subph=0
nxt_subph=0
TITLE=""

#
CPU_CORE=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $4}'`


# =========================
# FUNCTION
# =========================


function title {
	TITLE=$1 
	echo 
	echo "=================================================="
	echo -e "$HEAD_INFO: $TITLE"
	echo "=================================================="
	echo 
}

function error_exit {
	echo "$1" 1>&2
	exit 1
}

function get_phase {
	fn=$LOG_FN
	if [ -f $fn ]; then
		ph=`tail -n 1 $fn`
	else 
		echo "0 0" > $fn 
		ph="0 0"
	fi 
	#  return
	phase=`echo $ph | awk '{print $1}'`
	subph=`echo $ph | awk '{print $2}'`
}

function cfg_phase {
	ist_phase=$nxt_phase
	get_phase
	nxt_phase=`echo $((ist_phase+1))`
}

function cfg_subph {
	ist_subph=$nxt_subph
	get_phase
	nxt_subph=`echo $((ist_subph+1))`

	title "$TITLE -> $ist_subph"
}


function rm_f {
	fn=$1
	if [ -f $fn ]; then 
		rm -r $fn 
	fi 
}

function rm_d {
	dn=$1
	if [ -d $dn ]; then 
		rm -r $dn 
	fi 
}

function finish {
	get_phase
	if [ $subph -eq $nxt_subph ]; then 
		echo "$nxt_phase 0" >> $LOG_FN
	fi 
}

function finish_subph {
	cd $INSTALL_PATH
	if [ $subph -eq $ist_subph ]; then 
		echo "$phase $nxt_subph" >> $LOG_FN
	fi 

	if [ $phase -eq $PHASE_ED -a $subph -eq $SUBPH_ED ]; then 
		exit
	fi 
}


# =========================
# CHECK VERSION
# =========================


function normalize_version {
	version=$1
	echo $version | awk -F '.' '{print $1" "$2" "$3" x x x"}' | awk '{print $1"."$2"."$3}'
}

function match_version {
	name=$1 #
	version_target=$2 #target version
	version=$3 #version
	version_target=`normalize_version $version_target`
	version=`normalize_version $version`
    #
	RT_MATCH_VERSION=0
	if [ $version = "x.x.x" ]; then
		echo -e "$FLAG_FALSE $name is not installed"
		RT_MATCH_VERSION=0
	else
		if [ $version = $version_target ]; then
			echo -e "$FLAG_RIGHT $name==$version : Recommended version==$version_target "
			RT_MATCH_VERSION=1
		else
			echo -e "$FLAG_NRIGHT $name==$version : Recommended version==$version_target "
			RT_MATCH_VERSION=2
		fi
	fi 
}


function find_cmd {
	#find command, such as git, bazel
	cmd=$1
	RT_FIND_CMD=1
	which $cmd || RT_FIND_CMD=0
	if [ $RT_FIND_CMD -eq 0 ]; then 
		echo -e "${FLAG_FALSE} $cmd"
	else 
		echo -e "${FLAG_RIGHT} $cmd"
	fi 
}

function find_env {
	#find environment created by conda
	env=$1
	RT_FIND_ENV=1
	conda env list | grep "$env\ " || RT_FIND_ENV=0
	if [ $RT_FIND_ENV -eq 0 ]; then 
		echo -e "${FLAG_FALSE} $env"
	else 
		echo -e "${FLAG_RIGHT} $env"
	fi 
}

function pip_check_install {
	name=$1
	version_target=$2
	version=`pip list | grep "$name\ " | awk '{print $2}'`
	match_version $name $version_target $version
	if [ $RT_MATCH_VERSION -eq 0 ]; then
		echo "pip install $name==$version_target"
		pip install $name==$version_target || error_exit "$LINENO: failed" 

		version=`pip list | grep "$name\ " | awk '{print $2}'`
		match_version $name $version_target $version
	fi
}


# =========================
# INSTAL PHASE
# =========================

function install_dependency
{
	title "install dependency"

	# conda
	find_cmd conda
	if [ $RT_FIND_CMD -eq 0 ]; then
		echo "Please install conda"
		exit
	fi 

	# gcc
	version=`gcc --version | grep gcc | awk '{print $NF}'`
	match_version gcc $GCC_VERSION $version
	if [ $RT_MATCH_VERSION -ne 1 ]; then
		echo "Please install Recommended gcc==$GCC_VERSION"
		sleep 1
	fi 

	# git
	find_cmd git
	if [ $RT_FIND_CMD -eq 0 ]; then
		echo "Install git by conda"
		conda install git 
	fi 

	# env
	CONDA_ENV=`conda info | grep "active\ environment" | awk '{print $NF}'`
	if [ $CONDA_ENV = "None" ]; then
		echo "Please create a conda environment and activate it: such as"
		echo "\$ conda create -n nvnmd"
		echo "\$ conda activate nvnmd"
		exit
	fi
	ENV_PATH=`conda info | grep "active\ env\ location" | awk '{print $NF}'`
	echo "conda environment is : $CONDA_ENV"
	echo "environment location is : $ENV_PATH"

	# python
	# find_cmd python
	version=`python --version | awk '{print $NF}'`
	match_version python $PYTHON_VERSION $version
	if [ $RT_MATCH_VERSION -ne 1 ]; then
		echo "conda install python==$PYTHON_VERSION"
		conda install python==$PYTHON_VERSION || error_exit "$LINENO: failed"
		find_cmd python
	fi 
	version=`python --version | awk '{print $NF}'`
	match_version python $PYTHON_VERSION $version
}

function install_cmake
{
	title "install cmake"
	nxt_subph=0

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		rm_d $INSTALL_PATH/cmake  
		mkdir -p $INSTALL_PATH/cmake  
		cp $PACKAGE/$CMAKE.tar.gz cmake
		cd cmake
		tar -zxf $CMAKE.tar.gz 
		rm -r $CMAKE.tar.gz
		cd $CMAKE
		cp $PATCH/cmake/* .
	fi 
	finish_subph

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		cd cmake/$CMAKE 
		chmod +x ./bootstrap
		./bootstrap || error_exit "$LINENO: failed"
	fi 
	finish_subph

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		cd cmake/$CMAKE 
		make || error_exit "$LINENO: failed"
		sudo make install || error_exit "$LINENO: failed" 
	fi 
	finish_subph
}

function install_tensorflow {
	title "install tensorflow"
	nxt_subph=0

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		RT_FIND_TF=1
		ls $ENV_PATH/lib/libtensorflow_cc.so* || RT_FIND_TF=0
		if [ $RT_FIND_TF -eq 0 ]; then 
			echo -e "${FLAG_FALSE} have not installed libtensorflow_cc"
			echo "installed libtensorflow_cc"
			#conda search libtensorflow_cc -c deepmodeling
			conda install libtensorflow_cc=$LIBTF_CC_VERSION=$LIBTF_CC_BUILD -c deepmodeling || error_exit "$LINENO: failed"
		fi 
	fi 
	finish_subph

	cfg_subph
	if [ $phase -eq $ist_subph ]; then 
		pip_check_install tensorflow $TENSORFLOW_VERSION || error_exit "$LINENO: failed"
	fi 
	finish_subph
}

function install_nvnmd {
	title "install nvnmd"
	nxt_subph=0

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		pip_check_install matplotlib 3.5.1 || error_exit "$LINENO: failed" 
		pip_check_install ase 3.22.1 || error_exit "$LINENO: failed" 
	fi 
	finish_subph
	
	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		rm_d $INSTALL_PATH/$NVNMD_PATH  
		mkdir -p $INSTALL_PATH/$NVNMD_PATH  
		cp $PACKAGE/nvnmd.tar.gz $NVNMD_PATH 
		cd $NVNMD_PATH
		tar -zxf $NVNMD.tar.gz
		rm -r $NVNMD.tar.gz
		# patch
		if [ -d $PATCH/nvnmd ]; then 
			cp -r $PATCH/nvnmd/* nvnmd
		fi
	fi 
	finish_subph

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		cd $NVNMD_PATH/nvnmd
		cd ./source 
		rm_d build 
		mkdir build 
		cd build
		$CMAKE_BIN/cmake -DTENSORFLOW_ROOT=$ENV_PATH -DCMAKE_INSTALL_PREFIX=$NVNMD_BIN .. || error_exit "$LINENO: failed" 
		make -j $CPU_CORE || error_exit "$LINENO: failed" 
		make install || error_exit "$LINENO: failed" 
	fi 
	finish_subph

	cfg_subph
	if [ $subph -eq $ist_subph ]; then 
		cd $NVNMD_PATH/nvnmd/ 
		pip install . || error_exit "$LINENO: failed" 
		if [ -f "./deepmd/nvnmd/__main__.py" ]; then 
			echo "python $INSTALL_PATH/$NVNMD_PATH/nvnmd/deepmd/nvnmd/__main__.py \$@" > $ENV_PATH/bin/nvnmd
			chmod +x $ENV_PATH/bin/nvnmd
		fi 
	fi 
	finish_subph
}

function help {
	echo "bash install.sh [-i \"PHASE_ST SUBPH_ST PHASE_ED SUBPH_ED\"] "
	echo "                [-p NVNMD_PATH]"
	echo "PHASE:"
	echo "phase 0 : install cmake : 0-3"
	echo "phase 1 : install tensorflow and libtensorflow_cc : 0-2"
	echo "phase 2 : install nvnmd : 0-4"
	echo "phase 3 : finish : 0"
	exit 
}

# =========================
# MAIN
# =========================

get_phase

# get the parameter
while getopts 'i:p:' OPT; do 
case $OPT in 
i) 
PHASE_ST=`echo "$OPTARG" | awk '{print $1}'`
SUBPH_ST=`echo "$OPTARG" | awk '{print $2}'`
PHASE_ED=`echo "$OPTARG" | awk '{print $3}'`
SUBPH_ED=`echo "$OPTARG" | awk '{print $4}'`
;;
p)
NVNMD_PATH=`echo "$OPTARG" | awk '{print $1}'`
;;
*)
help 
;;
esac 
done 

echo ""
echo "$PHASE_ST $SUBPH_ST $PHASE_ED $SUBPH_ED $NVNMD_PATH"
echo "$PHASE_ST $SUBPH_ST" >> $LOG_FN
sleep 2

title "install "
echo "(1) install_dependency"
echo "(2) install_cmake"
echo "(3) install_tensorflow"
echo "(4) install_nvnmd"

cd $INSTALL_PATH
nxt_phase=0

# dependency
install_dependency

# cmake
cfg_phase
if [ $ist_phase -eq $phase ]; then 
	install_cmake
	finish
fi

# tensorflow
cfg_phase
if [ $ist_phase -eq $phase ]; then 
	install_tensorflow 
	finish
fi

# nvnmd
cfg_phase
if [ $ist_phase -eq $phase ]; then 
	install_nvnmd
	finish
fi

# finish
cfg_phase
if [ $ist_phase -eq $phase ]; then 
	echo -e "$HEAD_INFO: Finish"
fi

