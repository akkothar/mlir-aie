# Bottleneck B Implementation on AI Engine

## Overview

This project implements the Bottleneck B block of MobileNet V3 on AI Engine. In Bottleneck B, each bottleneck block is distributed across three AI cores, balancing computational load and parallelism to achieve efficient performance.

## Contents

- `README.md`: This file, providing an overview and setup instructions.


## Architecture

In Bottleneck B, each bottleneck block is divided and distributed across three AI cores. This design ensures efficient parallelism and load balancing, enhancing performance while maintaining the integrity of the MobileNet V3 architecture.

## Setup

### Building the Project

To compile and run the chained design:
```
cd bottleneck_B
make
```