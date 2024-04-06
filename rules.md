A::B is a system with 4 tokens: "A#", "#A", "B#" and "#B".
An A::B program is a sequence of tokens. Example:
B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

A# #A ... becomes ... nothing
A# #B ... becomes ... #B A#
B# #A ... becomes ... #A B#
B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each other,
they must be rewritten according to the corresponding rule. For example, the
first example shown here is computed as:

B# A# #B #A B# =
B# #B A# #A B# =
A# #A B# =

B#

The steps were:

1. We replaced "A# #B" by "#B A#".
2. We replaced "B# #B" by nothing.
3. We replaced "A# #A" by nothing.
The final result was just "B#".

Now, consider the following program:
A# B# B# #A B# #A #B

Fully compute it, step by step.
