Feature: Find the quantum number L_z
    As we want to be able to classify states
    We need to calculate the quantum number L_z

    Scenario: find the quantum number L_z of a single hydrogen atom
        When PyLATO is run using the job definition file: "scase_dimer.json"
        Then the quantum number L_z is 0

    @fixture.backup.TBcanonical_d.json
    Scenario: find the quantum number L_z of a pcase dimer with U/|t| = 2
        Given the TBcanonical_p model is set to have U/|t| = 2
        And the TBcanonical_p model is set to have 3 electrons
        When PyLATO is run using the job definition file: "pcase_dimer.json"
        Then the quantum number L_z is 2
