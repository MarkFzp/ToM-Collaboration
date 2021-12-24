import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

def repeat_tensor(tensor, repetion):
  exp_tensor = tf.expand_dims(tensor, axis=-1)
  tensor_t = tf.tile(exp_tensor, [1] + repetion)
  tensor_r = tf.reshape(tensor_t, repetion * tf.shape(tensor))
  return tensor_r


def get_ops(bad_mode, payoff_values, batch_size, num_hidden=32):
  # Input is a single number for agent 0 and agent 1.
  input_0 = tf.placeholder(tf.int32, shape=batch_size)
  input_1 = tf.placeholder(tf.int32, shape=batch_size)

  # Payoff matrix.
  num_cards = payoff_values.shape[0]     # C.
  num_actions = payoff_values.shape[-1]  # A.
  payoff_tensor = tf.constant(payoff_values)

  # Agent 0.
  with tf.variable_scope('agent_0'):
    weights_0 = tf.get_variable('weights_0', shape=(num_cards, num_actions))
    baseline_0_mlp = snt.nets.MLP([num_hidden, 1]) 

  # Agent 1.
  with tf.variable_scope('agent_1'):
    p1 = snt.nets.MLP([num_hidden, num_actions])
    baseline_1_mlp = snt.nets.MLP([num_hidden, 1])

  # These are the 'counterfactual inputs', i.e., all possible cards.
  all_inputs = tf.placeholder(tf.int32, shape=(1, num_cards))

  # We repeat them for each batch.
  repeated_in = tf.reshape(repeat_tensor(all_inputs, [batch_size, 1]), [-1])

  # Next we calculate the counterfactual action and log_p for each batch 
  # and hand.
  log_p0 = tf.nn.log_softmax(
      tf.matmul(tf.one_hot(repeated_in, num_cards), weights_0))
  cf_action = tf.to_int32(
      tf.squeeze(tf.multinomial(log_p0, num_samples=1)))  # [BC].
  
  # Produce log-prob of action selected.
  log_cf = tf.reduce_sum(
      log_p0 * tf.one_hot(cf_action, num_actions), axis=-1)  # [BC].

  # Some reshaping.
  repeated_in = tf.reshape(repeated_in, [batch_size, -1])  # [B,C].
  cf_action = tf.reshape(cf_action, [batch_size, -1])  # [B,C].
  log_cf = tf.reshape(log_cf, [batch_size, -1])  # [B,C].

  # Now we need to know the action the agent actually took. 
  # This is done by indexing the cf_action with the private observation.
  u0 = tf.reduce_sum(
      cf_action * tf.one_hot(input_0, num_cards, dtype=tf.int32), axis=-1)

  # Do the same for the log-prob.
  log_p0 = tf.reduce_sum(log_cf * tf.one_hot(input_0, num_cards), axis=-1)

  # Joint-action includes all the counterfactual probs - it's simply the sum.
  joint_log_p0 = tf.reduce_sum(log_cf, axis=-1)

  # Repeating the action chosen so that we can check all matches.
  repeated_actions = repeat_tensor(
      tf.reshape(u0, [batch_size, 1]), [1, num_cards])

  # A hand is possible iff the action in that hand matches the action chosen.
  weights = tf.to_int32(tf.equal(cf_action, repeated_actions))

  # Normalize beliefs to sum to 1.
  beliefs = tf.to_float(
      tf.divide(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))) 

  # Stop gradient mostly as a precaution.
  beliefs = tf.stop_gradient(beliefs)

  # Agent 1 receives beliefs + private ops for agent 1, unless it's 
  # the pure policy gradient version.
  if bad_mode == 2:
    joint_in1 = tf.concat([
        tf.one_hot(u0, num_actions, dtype=tf.float32), 
        tf.one_hot(input_1, num_cards, dtype=tf.float32),
    ], axis=1)
  else:
    joint_in1 = tf.concat([
        tf.one_hot(u0, num_actions, dtype=tf.float32),
        beliefs,
        tf.one_hot(input_1, num_cards, dtype=tf.float32),
    ], axis=1)
  joint_in1 = tf.reshape(joint_in1, [batch_size, -1])

  # We use a centralised baseline that contains both cards as input.
  baseline_0_input = tf.concat(
      [tf.one_hot(input_0, num_cards), tf.one_hot(input_1, num_cards)], axis=1) 
  baseline_1_input = tf.concat(
      [tf.one_hot(input_0, num_cards), joint_in1], axis=1)

  # Calculate baselines.
  baseline_0 = baseline_0_mlp(baseline_0_input)
  baseline_1 = baseline_1_mlp(baseline_1_input)

  # Giving the beliefs a fixed shape so that sonnet doesn't complain 
  # (probably there's a better way).
  beliefs = tf.reshape(beliefs, [batch_size, num_cards])

  # Evaluate policy for agent 1.
  log_p1 = tf.cast(tf.nn.log_softmax(p1(joint_in1)), tf.float32)

  # Sample agent 1 and get log-prob of action selected.
  u1 = tf.to_int32(tf.squeeze(tf.multinomial(log_p1, num_samples=1)))
  log_p1 = tf.reduce_sum(log_p1 * tf.one_hot(u1, num_actions), axis=-1)

  # Getting the rewards is just indexing into the payout matrix for all 
  # elements in the batch.
  rewards = tf.stack([
      payoff_tensor[input_0[i], input_1[i], u0[i], u1[i]] 
      for i in range(batch_size)
  ], axis=0)

  # Log-prob used for learning.
  if bad_mode == 1:
    log_p0_train = joint_log_p0
  else:
    log_p0_train = log_p0
  log_p1_train = log_p1

  # Policy-gradient loss.
  pg_final = tf.reduce_mean(
      (rewards - tf.stop_gradient(baseline_0)) * log_p0_train)
  pg_final += tf.reduce_mean(
      (rewards - tf.stop_gradient(baseline_1)) * log_p1_train)

  # Baseline loss.
  total_baseline_loss = tf.reduce_mean(tf.square(rewards - baseline_0)) 
  total_baseline_loss += tf.reduce_mean(tf.square(rewards - baseline_1)) 

  # Train policy.
  opt_policy = tf.train.AdamOptimizer()
  train_policy = opt_policy.minimize(-pg_final)
  
  # Train baseline.
  opt_baseline = tf.train.AdamOptimizer()
  train_baseline = opt_baseline.minimize(total_baseline_loss)
  
  # Pack up the placeholders.
  phs = {
      'input_0': input_0,
      'input_1': input_1,
      'all_inputs': all_inputs,
  }

  # Pack up the train ops.
  train_ops = {
      'policy': train_policy,
      'baseline': train_baseline,     
  }
  
  return rewards, train_ops, phs


def train(bad_mode,
          batch_size=32,
          num_runs=1,
          num_episodes=5000,
          num_readings=100,
          seed=42,
          debug=False):
  # Payoff values, [C,C,A,A].
  payoff_values = np.asarray([
      [
          [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
          [[0, 0, 10], [4, 8, 4], [0, 0, 10]],
      ],
      [
          [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
          [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
      ],
  ], dtype=np.float32)
  num_cards = payoff_values.shape[0]  # C.

  # All cards.
  all_cards = np.zeros((1, num_cards))
  for i in range(num_cards):
    all_cards[0, i] = i

  # Reset TF graph.
  tf.reset_default_graph()

  # Set random number generator seeds for reproducibility.
  tf.set_random_seed(seed)
  np.random.seed(seed)
  
  # Build graph.
  rewards_op, train_ops, phs = get_ops(bad_mode, payoff_values, batch_size)
  
  # Initializer.
  init = tf.global_variables_initializer()
  
  # Run.
  rewards = np.zeros((num_runs, num_readings + 1))
  interval = num_episodes // num_readings
  with tf.Session() as sess:      
    for run_id in range(num_runs):
      if run_id % max(num_runs // 10, 1) == 0:
        print('Run {}/{} ...'.format(run_id + 1, num_runs))
      
      sess.run(init)
      for episode_id in range(num_episodes + 1):
        cards_0 = np.random.choice(num_cards, size=batch_size)
        cards_1 = np.random.choice(num_cards, size=batch_size)

        fetches = [rewards_op, train_ops['baseline'], train_ops['policy']]
        feed_dict = {
            phs['input_0']: cards_0,
            phs['input_1']: cards_1,
            phs['all_inputs']: all_cards,
        }
        reward = sess.run(fetches, feed_dict)[:-2]  # Ignore train ops.
        reward = np.mean(reward)  # Average over batch.

        # Maybe save.
        if episode_id % interval == 0:
          rewards[run_id, episode_id // interval] = reward

        # Maybe log.
        if debug and episode_id % (num_episodes // 5) == 0:
          print(episode_id, 'reward:', reward)

  return rewards