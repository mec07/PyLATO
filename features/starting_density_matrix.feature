Feature: Input a density matrix as a starting point for the calculation
    As it is possible to get stuck in local minima in tight binding calculations
    We would like to be able to start the calculation with an inputted density matrix
    And this will help us to find an alternative solution to the tight binding calculation

    @fixture.backup.TBcanonical_s.json
    Scenario: regular calculation for scase dimer with U/|t| = 4
        Given the TBcanonical_s model is set to have U/|t| = 4
        When PyLATO is run using the job definition file: "scase_dimer.json"
        Then the total energy in the output file is 0

    @fixture.backup.TBcanonical_s.json
    Scenario: input a density matrix for the scase dimer with U/|t| = 4
        Given the TBcanonical_s model is set to have U/|t| = 4
        When PyLATO is run using the job definition file: "scase_dimer_input_density_matrix.json"
        Then the total energy in the output file is -0.5
