singularity run --cleanenv \
--bind /export/home/orenkobo/Aim3/dataset/narratives:/export/home/orenkobo/Aim3/dataset/narratives/ \
--bind /export/home/orenkobo/Aim3:/export/home/orenkobo/Aim3/ \
/export/home/orenkobo/fmriprep_singularity_vodkaintact/fmriprep_1.3.0.post2.simg \
fmriprep /export/home/orenkobo/Aim3/dataset/narratives/ /export/home/orenkobo/Aim2/vodka_intact_fmriprep/deriviatives participant -w /export/home/orenkobo/fmriprep_singularity_vodkaintact/jobs/work/work_$PartIDs --ignore slicetiming

if [ $? -eq "0" ]; then
rm -rf /export/home/orenkobo/fmriprep_singularity_vodkaintact/jobs/work/work_$PartIDs
fi

singularity run --cleanenv \
--bind /export/home/orenkobo/Aim2/BIDS/vodka-scramble/:/export/home/orenkobo/Aim2/BIDS/vodka-scramble/ \
--bind /export/home/orenkobo/Aim2:/export/home/orenkobo/Aim2 \
/export/home/orenkobo/fmriprep_singularity/fmriprep_1.3.0.post2.simg \
fmriprep /export/home/orenkobo/Aim2/BIDS/vodka-scramble/ /export/home/orenkobo/Aim2/BIDS/vodka-scramble/deriviatives/ participant -w /export/home/orenkobo/fmriprep_singularity/jobs/work/work_$PartIDs --ignore slicetiming

if [ $? -eq "0" ]; then
rm -rf /export/home/orenkobo/fmriprep_singularity/jobs/work/work_$PartIDs
fi