
for method in 'mult'
do
    for text_backbone in 'bert-base-uncased' 
    do
        for ood_dataset in  'MIntRec-OOD' 
        do
            for ood_detection_method in  'ma' 
            do
                CUDA_VISIBLE_DEVICES=0 python run.py \
                --dataset 'MIntRec' \
                --data_path '/root/autodl-tmp' \
                --ood_dataset $ood_dataset \
                --logger_name ${method}_${ood_detection_method} \
                --multimodal_method $method \
                --method ${method}\
                --ood_detection_method $ood_detection_method \
                --ood \
                --train \
                --pretrain \
                --tune \
                --save_results \
                --save_model \
                --gpu_id '0' \
                --video_feats_path 'swin_feats.pkl' \
                --audio_feats_path 'wavlm_feats.pkl' \
                --ood_video_feats_path 'swin_feats.pkl' \
                --ood_audio_feats_path 'wavlm_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name $method'_MIntRec' \
                --output_path '/root/autodl-tmp/output' \
                --results_file_name 'results1.csv' 
            done
        done
    done
done

