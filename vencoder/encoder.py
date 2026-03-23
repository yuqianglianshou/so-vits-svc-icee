from path_utils import get_contentvec_hf_path


class SpeechEncoder(object):
    def __init__(self, vec_path=None, device=None):
        self.model = None  # This is Model
        self.hidden_dim = 768
        self.vec_path = str(get_contentvec_hf_path()) if vec_path is None else vec_path
        pass


    def encoder(self, wav):
        """
        input: wav:[signal_length]
        output: embedding:[batchsize,hidden_dim,wav_frame]
        """
        pass
