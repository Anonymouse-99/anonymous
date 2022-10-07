echo 'Download the ISRUC-S3 dataset. (wget and unrar is needed)'
echo 'Make data dir: ./ISRUC_S3'
mkdir -p ./ISRUC_S3/ExtractedChannels
mkdir -p ./ISRUC_S3/RawData

cd ./ISRUC_S3/RawData
for s in {1..10}:
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
    unrar x $s.rar
done
echo 'Download Data to "./ISRUC_S3/RawData" complete.'

cd ./ISRUC_S3/ExtractedChannels
for s in {1..10}:
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat
done
echo 'Download ExtractedChannels to "./ISRUC_S3/ExtractedChannels" complete.'
