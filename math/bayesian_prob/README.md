### Bayesian Probability
<p>Bayesian probability is an interpretation of the concept of probability, in which, instead of frequency or propensity of some phenomenon, probability is interpreted as reasonable expectation representing a state of knowledge or as quantification of a personal belief.</p>

<p>Bayesian probability belongs to the category of evidential probabilities; to evaluate the probability of a hypothesis, the Bayesian probabilist specifies some <em>prior probability</em>, which is then <em>updated</em> to a <em>posterior probability</em> in the light of new, relevant data (evidence). The Bayesian interpretation provides a standard set of procedures and formulae to perform this calculation.</p>

 #### Likelihood 
<p>The likelihood function (often simply called the likelihood) is a function of the parameters of a statistical model. Likelihood functions play a key role in statistical inference, especially methods of estimating a parameter from a set of statistics. In informal contexts, "likelihood" is often used as a synonym for "probability."</p>

<p>Given some data, the likelihood of some hypothesis is proportional to the probability of observing that data given that hypothesis. It is a function of the parameters of the hypothesis, but viewed as a function of the observed data, with the parameters held fixed.</p>

<p>For example, consider a set of data drawn from a normal distribution with unknown mean and variance. The likelihood of a given hypothesis (i.e., a particular pair of values for the mean and the variance) is proportional to the probability of observing the given data given that hypothesis. The likelihood function is a function of the parameters of the hypothesis, but viewed as a function of the observed data, with the parameters held fixed.</p>

```python
def likelihood(data, hypo):
    """Computes the likelihood of the data under the hypothesis.

    hypo: integer value of lambda, hypothetical number of arrivals
    data: list of interarrival times in seconds

    returns: float probability
    """
    lam = hypo
    likelihood = 1
    for x in data:
        likelihood *= thinkbayes2.EvalPoissonPmf(lam, x)
    return likelihood
```

 #### Posterior distribution
<p>In Bayesian statistics, a <em>posterior probability distribution</em>, often simply called the <em>posterior</em>, of an uncertain quantity is the conditional probability distribution of the quantity given the observed data. For example, suppose the quantity is the parameter of a parametric family of probability distributions. The <em>data</em> may be a set of observations of instances of the phenomenon being studied, or the data may be a summary of existing data in the form of a set of sufficient statistics.</p>

```python
def main():
    hypos = range(0, 151)
    suite = thinkbayes2.Suite(hypos)
    suite.Update((60, 30, 90))
    thinkplot.Pmf(suite)
    thinkplot.Config(xlabel='Number of trains',
                     ylabel='Probability')
    print('Maximum Likelihood', suite.MaximumLikelihood())
    print('Mean', suite.Mean())
    print('Median', thinkbayes2.Percentile(suite, 50))
    print('CI', thinkbayes2.CredibleInterval(suite, 90))
    print('Posterior mean', suite.Mean())
    print('Posterior median', thinkbayes2.Percentile(suite, 50))
    print('Posterior credible interval', thinkbayes2.CredibleInterval(suite, 90))
    thinkplot.Pmf(suite)
    thinkplot.Config(xlabel='Number of trains',
                     ylabel='Probability')
    thinkplot.Pmf(suite.MakePmfFromCdf())
    thinkplot.Config(xlabel='Number of trains',
                     ylabel='Probability')
    thinkplot.Cdf(suite.MakeCdf())
    thinkplot.Config(xlabel='Number of trains',
                     ylabel='Probability')
    thinkplot.Show()
```
formula:

P(H|D) = P(D|H) * P(H) / P(D)

P(H|D) = posterior probability of H given D
P(D|H) = likelihood of D given H
P(H) = prior probability of H
P(D) = prior probability of D - marginal likelihood/normalizing constant


