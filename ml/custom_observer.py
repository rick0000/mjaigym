from typing import List

import numpy as np

from .framework import MjObserver
from .features import *

class SampleCustomObserver(MjObserver):
    """MjObserverを実装した例
    特徴量抽出クラスであるFeaturesクラス、FuroAppendFeatureクラスを使って特徴量を生成している。
    自分の考えた特徴量を追加したい場合はFeaturesクラス、FuroAppendFeatureクラスを元に
    新たな特徴量抽出クラスを作成しOnTsumoFeaturesまたはOnOtherDahaiFeaturesに追加すると良い。
    """

    OnTsumoFeatures:List[Feature] = [
            AnkanFeature,
            AnpaiFeature,
            BakazeFeature,
            ChiFeature,
            DiscardFeature,
            DiscardReachStateFeature,
            DiscardRedPaiFeature,
            DoraFeature,
            FuroRedDoraFeature,
            JikazeFeature,
            KyokuInfoFeature,
            LastDahaiFeature,
            MinkanFeature,
            OyaFeature,
            PointFeature,
            PonFeature,
            PlayerRedDoraFeature,
            ReachFeature,
            RestTsumoNumFeature,
            RestPaiInViewFeature,
            TehaiFeature,
            TypeFeature,
            # HorapointDfsFeature,
        ]
    OnTsumoFeaturesLength = sum([f.get_length() for f in OnTsumoFeatures])

    OnOtherDahaiFeatures:List[FuroAppendFeature] = [
            FuroCandidateFuroAppendFeature,
            ShantenFuroAppendFeature,
        ]
    OnOtherDahaiFeaturesLength = sum([f.get_length() for f in OnOtherDahaiFeatures])


    def get_tsumo_observe_channels_num(self):
        return self.OnTsumoFeaturesLength

    def get_otherdahai_observe_channels_num(self):
        return self.OnOtherDahaiFeaturesLength

    def trainsform_dahai(self, state, id, oracle_enable_flag):
        return self._calc_on_tsumo_feature(state, id, oracle_enable_flag)

    def trainsform_reach(self, state, id, oracle_enable_flag):
        return self._calc_on_tsumo_feature(state, id, oracle_enable_flag)

    def trainsform_chi(self, state, id, candidate_action, oracle_enable_flag):
        return self._calc_on_other_dahai_feature(state, id, candidate_action, oracle_enable_flag)

    def trainsform_pon(self, state, id, candidate_action, oracle_enable_flag):
        return self._calc_on_other_dahai_feature(state, id, candidate_action, oracle_enable_flag)

    def trainsform_kan(self, state, id, candidate_action, oracle_enable_flag):
        return self._calc_on_other_dahai_feature(state, id, candidate_action, oracle_enable_flag)


    def _calc_on_tsumo_feature(self, state, id, oracle_enable_flag):
        feature_area = np.zeros((self.OnTsumoFeaturesLength, 34, 1), dtype='int8')
        start_index = 0
        for feature in self.OnTsumoFeatures:
            feature_length = feature.get_length()
            target_feature_area = feature_area[start_index:start_index+feature_length]
            feature.calc(target_feature_area, state, id, oracle_enable_flag)
            start_index += feature_length
        return feature_area

    def _calc_on_other_dahai_feature(self, state, id, candidate_action, oracle_enable_flag):
        base_feature = self._calc_on_tsumo_feature(state, id, oracle_enable_flag)
        
        feature_area = np.zeros((self.OnOtherDahaiFeaturesLength, 34, 1), dtype='int8')
        start_index = 0
        for feature in self.OnOtherDahaiFeatures:
            feature_length = feature.get_length()
            target_feature_area = feature_area[start_index:start_index+feature_length]
            feature.calc(result=target_feature_area, board_state=state, player_id=id, candidate_furo=candidate_action)
            start_index += feature_length
        
        furo_appended = np.concatenate([base_feature, feature_area], axis=0)
        return furo_appended


