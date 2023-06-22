import time

def main():

    n = int(time.time())

    args = \
        ['--logdir',                        f'/home/cd/src/aal/logdir/crafter-dv3-20230502-{n}',
         '--env.crafter.outdir',            f'/home/cd/src/aal/logdir/crafter-dv3-20230502-{n}',
         #'--configs',                       'crafter',
         '--configs',                       'crafter', 'small',
         #'--configs',                       'dmc_vision',
         #'--task',                          'cdmc_cartpole_swingup',
         '--jax.jit',                       'True',
         '--replay',                        'curious-replay', # curious-replay; per; count-based; adversarial
         #'--replay',                        'per',
         '--replay_hyper.initial_priority', '1e5',
         '--replay_hyper.c',                '1e4',
         '--replay_hyper.beta',             '0.7',
         '--replay_hyper.epsilon',          '0.01',
         '--replay_hyper.alpha',            '0.7',
         # '--run.script',                    'train_eval',
         # '--run.steps',                     '1.5e4',
         # '--run.eval_every',                '1e4',
         # '--run.eval_initial',              'False',
         # '--run.eval_eps',                  '100',
         '--envs.amount',                     '1',
         #'--batch_size',                      '8',
         ]

    print('Local launch of Dreamer v3 🚀...')
    from dreamerv3 import train
    train.main(args)


if __name__ == '__main__':
    main()
