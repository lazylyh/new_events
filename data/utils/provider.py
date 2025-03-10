import abc


class DatasetProviderBase(abc.ABC):
    @abc.abstractmethod
    def get_predict_dataset(self):
        raise NotImplementedError

    
    @abc.abstractmethod
    def get_nbins_context(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_nbins_correlation(self):
        raise NotImplementedError