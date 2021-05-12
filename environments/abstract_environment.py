

class AbstractEnvironment(object):
    def __init__(self):
        pass

    def sample_data(self, n):
        """
        samples n random data points from environment, each of which is a tuple
            containing context and corresponding cost vector
        :return: tuple (x, y), where x is of shape (n, context_dim),
            and y is of shape (n, decision_dim)
        """
        return NotImplementedError()

    def get_constraints(self):
        """
        :return: constraints for given environment
        """
        return NotImplementedError()

    def get_context_dim(self):
        """
        :return: context_dim for given environment
        """
        return NotImplementedError()

    def get_decision_dim(self):
        """
        :return: decision_dim for given environment
        """
        return NotImplementedError()
