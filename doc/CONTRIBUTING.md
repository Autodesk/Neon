# Neon's Community and How to Contribute

Neon team strongly believe in the open source effort and we welcome and greatly appreciates contributions from the community: proposing new features, filing issues,  and contributing code. The rest of the document describe various processes to give your contribution. 

## Providing Suggestions

Neon is meant to evolve with feedback from the community, and we greatly appreciate any thoughts on ways to improve the design or features. Please use the `enhancement` tag to denote issues that are suggestions specifically - this helps us triage and respond appropriately.

## Filing Bugs

As with all software, you may run into bugs. Please submit bugs as regular issues on GitHub - we are regularly monitoring issues and will prioritize and schedule fixes.

The best bug reports include a detailed way to reproduce the issue predictably and possibly even a working example demonstrating the issue.

## Contributing Code

There are three main steps for contributing your code to the project. First, sign a Contributor License Agreement, share your goal with the community, write your code following Neon coding standards, and finally, submit a pull request. Here are some more details on each of the steps.


### Contributor License Agreement (CLA)

Before contributing any code to this project, we kindly ask you to sign a Contributor License Agreement (CLA). We can not accept any PR if a CLA has not been signed.

- If you are an individual contributor, the CLA signature is included as a part of the PR process.

- Please sign the document [ADSK-Form-Corp-Contrib-Agmt-for-Open-Source.pdf](./cla/ADSK-Form-Corp-Contrib-Agmt-for-Open-Source.pdf). The document includes instructions on where to send the completed forms to. Once a signed form has been received, we can happily review and accept your pull requests.

### Coordinate with the community

We highly recommend opening an issue on GitHub to describe your goals before starting any coding. This will allow you to get early feedback and avoid multiple parallel efforts.

### Coding Standard

To provide a more uniform code base, we'd appreciate it if any new code could follow the coding standard described in this document: [CodeConvention.md](CodeConvention.md).

### Git Workflow

We follow the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. 
If you'd like to contribute your code to Neon, you should:
- Include your work in a feature branch created from the Neon `develop` branch.
That branch contains the latest work in Neon. 
- Then, create a pull request with the `develop` branch as the destination.

Periodically, we merge the develop branch into the `main` branch and tag a new release.

When contributing code, please also include appropriate tests as part of the pull request, and follow the same comment and coding style as the rest of the project. Take a look through the existing code for examples of the testing and style practices the project follows.

