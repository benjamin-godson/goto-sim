"""
A test module that tests your example module.

Some people prefer to write tests in a test file for each function or
method/ class. Others prefer to write tests for each module. That decision
is up to you. This test example provides a single test for the example.py
module.
"""

from goto_sim.sim import GOTONode

def test_create_node():
    """
    Test that add_numbers works as expected.

    A single line docstring for tests is generally sufficient.
    """
    node = GOTONode(site='goto-n0rth', name='Test Node')
    assert node is not None

