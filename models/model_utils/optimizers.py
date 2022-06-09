from ml_collections import ConfigDict

import optax

def get_optimizer(config: ConfigDict):
    if config.optimizer_config.get('optimizer') == 'adam':
        return optax.adam(config.optimizer_config.get('base_learning_rate'),
                          config.optimizer_config.get('beta')[0],
                          config.optimizer_config.get('beta')[1],
                          config.optimizer_config.get('epsilon'),
                          config.optimizer_config.get('epsilon_root'),
                          config.optimizer_config.get('dtype'))

    if config.optimizer_config.get('optimizer') == 'adamw':
        return optax.adamw(config.optimizer_config.get('base_learning_rate'),
                          config.optimizer_config.get('beta')[0],
                          config.optimizer_config.get('beta')[1],
                          config.optimizer_config.get('epsilon'),
                          config.optimizer_config.get('epsilon_root'),
                          config.optimizer_config.get('dtype'),
                          config.optimizer_config.get('weight_decay'),
                          config.optimizer_config.get('mask'))

    if config.optimizer_config.get('optimizer') == 'sgd':
        return optax.sgd(config.optimizer_config.get('base_learning_rate'),
                         config.optimizer_config.get('momentum'),
                         config.optimizer_config.get('nesterov'),
                         config.optimizer_config.get('dtype'))

    if config.optimizer_config.get('optimizer') == 'rmsprop':
        return optax.rmsprop(config.optimizer_config.get('base_learning_rate'),
                             config.optimizer_config.get('decay'),
                             config.optimizer_config.get('epsilon'),
                             config.optimizer_config.get('initial_scale'),
                             config.optimizer_config.get('centered'),
                             config.optimizer_config.get('momentum'),
                             config.optimizer_config.get('nesterov')
                             )
