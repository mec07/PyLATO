Feature: Find the groundstate energy
    As there are analytical solutions to some tight binding problems
    We can use these to make sure that PyLATO numerically finds the correct solution
    And this will help to ensure that we do not break PyLATO while developing it

    Scenario: find the groundstate energy of a single hydrogen atom
        When PyLATO is run using the single hydrogen job definition file
        Then the total energy in the output file is -13.4
