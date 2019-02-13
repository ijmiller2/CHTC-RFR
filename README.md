# CHTC-RFR
Random forest regression for Jesse's K99 app

## Submit the interactive job to compile Python 3.6 from source

Notice the requirements line "requirements = (OpSysMajorVer =?= 7)". This means we want the operating system to CentOS 7, instead of Scientific Linux 6, which would be specified with "requirements = (OpSysMajorVer =?= 6)". The reason we want Centos 7 is because the high memory computer (RAM >= 128 GB) are all running this version of Linux and if we compile on the version of the operating system our Python code won't work. I learned both of these things the hard way.

So here's how you kick up an interactive job to compile Python from source.

```bash
condor_submit -i interactive_compile.sub 
```

Once your inside the compile node, you should get message that looks something like:
```bash
Submitting job(s).
1 job(s) submitted to cluster 125114902.
Waiting for job to start...
Welcome to slot1_3@matlab-build-2.chtc.wisc.edu!
```
Once, you inside you can pull the Python source code from [here]() with the `wget` command.
```bash
wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
```
Unpack the "tar ball":
```bash
tar zxvf Python-3.6.8.tgz
```

Then follow instructions from the [CHTC docs](http://chtc.cs.wisc.edu/python-jobs.shtml):
```bash
mkdir python
cd Python-3.6.8
./configure --prefix=$(pwd)/../python
make
make install
cd ..
ls python
ls python/bin
cp python/bin/python3 python/bin/python
cp python/bin/pip3 python/bin/pip
export PATH=$(pwd)/python/bin:$PATH 
which pip
pip install sklearn
tar -czvf python.tar.gz python/
exit
```

This should do it to get the compiled Python code back to the submit node. Now you just to unpackage this and then repackage it with the other needed input files before submitting a normal batch job.

```bash
cd input
tar zcvf combined.tar.gz python genMGP_trainRF1.py VCAP_6h5uM_8496x979_plusFP_4RF.csv
```

## Submit command
```bash
cd /home/ijmiller2/random_forest_regression/CHTC-RFR/condor_files
condor_submit sklearn_test.sub 
```

