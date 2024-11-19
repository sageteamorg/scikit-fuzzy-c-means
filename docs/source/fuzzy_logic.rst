Fuzzy Logic
===========

What is Fuzzy Logic?
--------------------

Fuzzy logic is a system of logic that deals with approximate reasoning, rather than precise and absolute reasoning. In classical logic, statements are either true or false. In fuzzy logic, the truth value of a statement can range between `0` and `1`, allowing for a more realistic way of dealing with uncertainty.

Basic Concepts of Fuzzy Logic
-----------------------------

**Classical Logic**

Classical logic, also known as Boolean logic, is based on binary truth values. Each proposition can either be true (`1`) or false (`0`). For instance:

- **Example**: "The light is on." 

  In classical logic, this statement can only be either `true` or `false`.

**Fuzzy Logic**

In fuzzy logic, a statement can have a truth value anywhere between `0` and `1`. This type of reasoning is better suited to handling situations where the boundaries of truth are not clearly defined. For instance:

- **Example**: "The room is warm."

  Instead of being strictly `true` or `false`, fuzzy logic allows the statement to have a truth value such as `0.7`, indicating that the room is moderately warm.

**Example**: Tallness

Consider the question of whether someone is tall. In classical logic, a strict height threshold may be defined. In fuzzy logic, height can be expressed with degrees of membership:

- A person of height `170 cm` might have a membership degree of `0.5` in the "tall" category.
- A person of height `190 cm` might have a membership degree of `0.9` in the "tall" category.

Membership Functions
--------------------

Membership functions are used in fuzzy logic to represent the degree to which an element belongs to a fuzzy set. Below are some common types of membership functions:

**Triangular Membership Function**

The triangular membership function is a simple and commonly used function shaped like a triangle. It is defined by three parameters: `a`, `b`, and `c`, which represent the left endpoint, the peak, and the right endpoint, respectively.

.. math::

    \mu(x) = \max\left(\min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right), 0\right)

**Trapezoidal Membership Function**

The trapezoidal membership function is a generalized version of the triangular membership function. It is defined by four parameters: `a`, `b`, `c`, and `d`, which form the shape of a trapezoid.

.. math::

    \mu(x) = \max\left(\min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right), 0\right)

**Gaussian Membership Function**

The Gaussian membership function has a smooth bell shape and is commonly used in fuzzy systems. It is defined by a mean value (`c`) and a standard deviation (`\sigma`).

.. math::

    \mu(x) = \exp\left(-\frac{(x - c)^2}{2\sigma^2}\right)

Fuzzy Rules
-----------

Fuzzy rules are "if-then" statements that define how to make decisions in a fuzzy system.

**Example Rules**

- **Rule 1**: *If the temperature is high, then the fan speed is fast.*
  
- **Rule 2**: *If the distance is short, then the braking force is strong.*

These rules allow the system to infer outputs from inputs using approximate reasoning.

Operations in Fuzzy Logic
-------------------------

Fuzzy logic involves operations that are analogous to set operations in classical logic but adapted to handle degrees of membership:

**Fuzzy AND** (Intersection)

The fuzzy AND operation, also known as intersection, takes the minimum of the involved membership values.

.. math::

    \mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))

**Fuzzy OR** (Union)

The fuzzy OR operation, also known as union, takes the maximum of the involved membership values.

.. math::

    \mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))

**Fuzzy NOT** (Complement)

The fuzzy NOT operation, also known as complement, is defined as one minus the membership value.

.. math::

    \mu_{\neg A}(x) = 1 - \mu_A(x)

Fuzzy Inference Process
-----------------------

The fuzzy inference process is used to derive a conclusion from a set of fuzzy rules. It consists of the following steps:

1. **Fuzzification**: Convert crisp inputs into fuzzy values using appropriate membership functions.
  
2. **Rule Evaluation**: Apply the fuzzy rules to determine the output fuzzy sets.

3. **Aggregation of Outputs**: Combine the output fuzzy sets from all rules into a single fuzzy set.

4. **Defuzzification**: Convert the fuzzy output back into a crisp value, often using methods such as `centroid` or `mean of maxima`.

**Example of Fuzzy Inference**

Consider a fuzzy system that controls the speed of a fan based on temperature:

- **Step 1**: Fuzzify the input temperature to determine its degree of membership in fuzzy sets like "low", "medium", or "high".
  
- **Step 2**: Apply rules such as "If temperature is high, then fan speed is fast".

- **Step 3**: Aggregate the output fuzzy sets from all applicable rules.

- **Step 4**: Defuzzify the aggregated result to get the actual speed of the fan (e.g., `75%` of maximum speed).

Summary
-------

Fuzzy logic is well-suited for situations that involve uncertainty or partial truths. Unlike classical logic systems, where outcomes are binary (either true or false), fuzzy systems provide a more nuanced approach, allowing for degrees of truth. This makes fuzzy logic especially powerful for applications such as control systems, pattern recognition, and decision-making in uncertain environments.
