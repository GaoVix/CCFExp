

def make_condition(pl_module, condition_type, condition_source, batch):

    result = {'cross_attn': None, 'add': None, 'center_emb': None}

    if condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        assert 'id_image' in batch
        id_feat, id_cross_att = pl_module.label_mapping(batch['id_image'])
        B = batch['image'].shape[0]
        face_f = pl_module.face_model(batch['image'].to(pl_module.device)) # deviation
        id_f = pl_module.id_model(batch['image'].to(pl_module.device))
        emotion = face_f - id_f # [B, 512]
        emotion = emotion.unsqueeze(1)
        emotion = pl_module.emotion_adapter(emotion) # [B, 49, 512]
        landmark_features = pl_module.landmark_detector(batch['image'].to(pl_module.device), sign='return_feature').view(B, 49, -1) # [B, 49, 512]
        cross_attn = pl_module.condition_adapter(id_cross_att, emotion, landmark_features) # [B, 512, 147]
        result['cross_attn'] = pl_module.cross_attn_adapter(cross_attn) # [B, 1024, 147]
        result['stylemod'] = id_feat
        # class_label = batch['class_label'].to(pl_module.device)
        # center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)
    else:
        raise ValueError('make Condition')

    # if center_emb.shape[1] == 1:
    #     center_emb = center_emb.squeeze(1)

    # result['center_emb'] = center_emb
    return result

def split_label_spatial(condition_type, condition_source, encoder_hidden_states, pl_module=None):

    if condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        num_label_features = pl_module.label_mapping.pos_emb[0].shape[0] - 1
        label_feat = encoder_hidden_states['stylemod']
        label_spat = encoder_hidden_states['cross_attn'][:, :, :num_label_features]
        spatial = encoder_hidden_states['cross_attn'][:, :, num_label_features:]
        label = [label_feat, label_spat]
    else:
        raise ValueError('not implemented')

    return label, spatial
