Feature: Find the groundstate energy
    As there are analytical solutions to some tight binding problems
    We can use these to make sure that PyLATO numerically finds the correct solution
    And this will help to ensure that we do not break PyLATO while developing it

    Scenario: find the groundstate energy of a single atom in a repeating cell
        Given there is a specification file for a single atom in a repeating cell
        When PyLATO is run
        Then the energy in the output file is ...
