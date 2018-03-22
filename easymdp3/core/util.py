from __future__ import division
import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)
#================================#
#
#    Functions related to policies
#    and probabilities
#
#================================#
np.seterr(all='raise')

def calc_softmax_dist(action_vals, temp=1.0):
    #normalization trick
    mval = max(action_vals.values())
    action_vals = {a: v - mval for a, v in action_vals.iteritems()}

    aprobs = {}
    norm = 0
    for a, q in action_vals.items():
        try:
            p = np.exp(q/temp)
        except FloatingPointError:
            p = 0
            warnings.warn(("Softmax underflow (q = %g, temp = %g); " +
                          "setting prob to 0.") % (q, temp))
        norm += p
        aprobs[a] = p
    aprobs = {a: p/norm for a, p in aprobs.items()}
    return aprobs

def calc_softmax_policy(stateaction_vals, temp=1):
    soft_max_policy = {}
    for s, a_q in stateaction_vals.iteritems():
        soft_max_policy[s] = calc_softmax_dist(a_q, temp=temp)
    return soft_max_policy

def calc_esoftmax_dist(a_vals, temp=0.0, randchoose=0.0):
    """
    See work by Nassar & Frank (2016) and Collins and Frank (2018)

    http://ski.clps.brown.edu/papers/NassarFrank_curopin.pdf and
    http://ski.clps.brown.edu/papers/CollinsFrank_PNAS_supp.pdf
    """
    if temp == 0.0:
        maxval = max(a_vals.values())
        maxacts = [a for a, v in a_vals.items() if v == maxval]
        act_randchoose = randchoose/len(a_vals)
        act_maxchoose = (1-randchoose)/len(maxacts)
        a_p = {}
        for a in a_vals.keys():
            a_p[a] = act_randchoose
            if a in maxacts:
                a_p[a] += act_maxchoose
    else:
        sm = calc_softmax_dist(a_vals, temp)
        act_randchoose = randchoose/len(a_vals)
        a_p = {}
        for a, smp in sm.items():
            a_p[a] = act_randchoose + (1 - randchoose)*smp
    return a_p

def calc_esoftmax_policy(sa_vals, temp=0.0, randchoose=0.0):
    policy = {}
    for s, a_q in sa_vals.iteritems():
        policy[s] = calc_esoftmax_dist(a_q, temp=temp, randchoose=randchoose)
    return policy

def calc_stochastic_policy(sa_vals, rand_choose=0.0):
    return calc_esoftmax_policy(sa_vals, temp=0.0, randchoose=rand_choose)

def calc_egreedy_dist(action_vals, rand_choose=0.0):
    return calc_esoftmax_dist(action_vals, randchoose=rand_choose, temp=0.0)

def sample_prob_dict(pdict):
    outs, p_s = zip(*pdict.items())
    out_i = np.random.choice(range(len(outs)), p=p_s)
    return outs[out_i]


def calc_traj_probability(policy, traj, get_log=False):
    if get_log:
        prob = 0
        for s, a in traj:
            prob += np.log(policy[s][a])
        return prob
    else:
        prob = 1
        for s, a in traj:
            prob *= policy[s][a]
        return prob

def argmax_dict(mydict, return_all_maxes=False, return_as_list=False):
    max_v = -np.inf
    max_k = []
    for k, v in mydict.iteritems():
        if v > max_v:
            max_v = v
            max_k = [k, ]
        elif v == max_v:
            max_k.append(k)

    if len(max_k) > 1:
        if return_all_maxes:
            return max_k
        else:
            if return_as_list:
                return [np.random.choice(max_k), ]
            else:
                return np.random.choice(max_k)
    else:
        if return_as_list:
            return max_k
        else:
            return max_k[0]
