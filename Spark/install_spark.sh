#!/bin/bash

# move to the root directory
cd ~

# # install java
sudo apt-get install openjdk-8-jdk scala || brew cask install openjdk-8-jdk scala


# # install spark
wget https://apache.osuosl.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
tar -zxf spark-3.1.2-bin-hadoop3.2.tgz
rm spark-3.1.2-bin-hadoop3.2.tgz
mv spark-3.1.2-bin-hadoop3.2 ~/spark

if [ -f ".zshrc" ]; then
  echo "you use zsh"
  PATH_FILE=.zshrc 

elif [ -f ".bash_profile" ]; then
  echo "found .bash_profile"
  PATH_FILE=.bash_profile
  
elif [ -f ".bashrc" ]; then
  echo "found .bashrc"
  PATH_FILE=.bashrc

elif [ -f ".profile" ]; then
  echo "found .profile"
  PATH_FILE=.profile
else
  echo "Could not find a bash file, PATH WAS NOT CHANGED"
  PATH_FILE=/dev/null
fi

# edit the path 
echo "# add spark path to PATH" >> $PATH_FILE
echo "export SPARK_HOME=~/spark" >> $PATH_FILE
echo "export PATH=\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin" >> $PATH_FILE
# echo "export PYSPARK_DRIVER_PYTHON=\"jupyter\"" >> $PATH_FILE
# echo "export PYSPARK_DRIVER_PYTHON_OPTS=\"notebook\"" >> $PATH_FILE

# source the path file
if [ PATH_FILE != "/dev/null" ]; then
  source $PATH_FILE
fi 

# get pyspark
pip install pyspark
