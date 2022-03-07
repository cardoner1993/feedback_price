import pandas as pd
import torch
from tqdm import tqdm


def inference(batch, cfg, model, ids_to_labels):
                
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(cfg['device'])
    mask = batch["attention_mask"].to(cfg['device'])
    outputs = model(ids, mask=mask)  # return_dict=False by default
    
    if cfg['build_custom_head']:
        all_preds = torch.argmax(outputs, axis=-1).cpu().numpy() 
    else:
        all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
    
    return predictions


# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# code has been modified from original
def get_predictions(df, loader, model, cfg):
    
    # put model in training mode
    model.eval()
    
    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    print("Builiding word label predictions")
    for batch in tqdm(loader, total=len(loader)):
        labels = inference(batch, cfg, model, cfg['ids_to_labels'])
        y_pred2.extend(labels)

    final_preds2 = []
    print("Building final preds")
    for i in tqdm(range(len(df)), total=len(df)):

        idx = df.id.values[i]
        #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i] # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B','I') # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            
            if cls != 'O' and cls != '' and end - j > 7:
                final_preds2.append((idx, cls.replace('I-',''),
                                     ' '.join(map(str, list(range(j, end))))))
        
            j = end
        
    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id','class','predictionstring']

    return oof


# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score
