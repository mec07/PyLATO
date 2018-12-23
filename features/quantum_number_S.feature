Feature: Find the quantum number S
    As we want to be able to classify states
    We need to calculate the quantum number S

    Scenario: find the quantum number S of a single hydrogen atom
        When PyLATO is run using the job definition file: "single_hydrogen_job_def.json"
        Then the quantum number S is 0.5

    @fixture.backup.TBcanonical_s.json
    Scenario: find the quantum number S of an scase dimer with U/|t| = 4
        Given the TBcanonical_s model is set to have U/|t| = 4
        When PyLATO is run using the job definition file: "scase_dimer.json"
        Then the quantum number S is 0
