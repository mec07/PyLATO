Feature: Find the wavefunction classification
    As we want to classify wavefunctions of dimers
    We need to calculate the classification

    @fixture.backup.TBcanonical_s.json
    Scenario: find the groundstate classification of an scase dimer
        When PyLATO is run using the job definition file: "scase_dimer.json"
        Then the groundstate classification is "{}^1\Sigma^{+}_{g}"

    @fixture.backup.TBcanonical_p.json
    Scenario: find the groundstate classification of a pcase dimer
        Given the TBcanonical_p model is set to have U/|t| = 4
        When PyLATO is run using the job definition file: "pcase_dimer.json"
        Then the groundstate classification is "{}^1\Sigma^{+}_{g}"

    @fixture.backup.TBcanonical_d.json
    Scenario: find the groundstate classification of a dcase dimer
        Given the TBcanonical_d model is set to have U/|t| = 2
        When PyLATO is run using the job definition file: "dcase_dimer.json"
        Then the groundstate classification is "{}^1\Sigma^{+}_{g}"
