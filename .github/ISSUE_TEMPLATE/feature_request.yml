name: 💡 Feature Request
description: Suggest an idea or enhancement for DmDSLab
title: "[FEATURE] "
labels: ["enhancement", "triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature! 🌟
        
        Please provide as much detail as possible to help us understand your needs.
        
  - type: textarea
    id: problem
    attributes:
      label: 🤔 Problem Description
      description: Is your feature request related to a problem? Please describe.
      placeholder: I'm always frustrated when...
    validations:
      required: true
      
  - type: textarea
    id: solution
    attributes:
      label: 💡 Proposed Solution
      description: Describe the solution you'd like to see
      placeholder: I would like to be able to...
    validations:
      required: true
      
  - type: textarea
    id: alternatives
    attributes:
      label: 🔄 Alternatives Considered
      description: Describe any alternative solutions or features you've considered
      placeholder: I also considered...
      
  - type: dropdown
    id: feature_type
    attributes:
      label: 🏷️ Feature Type
      description: What type of feature is this?
      options:
        - New dataset source integration
        - Data processing enhancement
        - New data container functionality
        - CLI tool improvement
        - Performance optimization
        - Documentation improvement
        - Testing enhancement
        - Other
    validations:
      required: true
      
  - type: dropdown
    id: priority
    attributes:
      label: ⚡ Priority
      description: How important is this feature to you?
      options:
        - Low - Nice to have
        - Medium - Would improve my workflow
        - High - Blocking my use case
        - Critical - Essential for my project
    validations:
      required: true
      
  - type: textarea
    id: use_case
    attributes:
      label: 🎯 Use Case
      description: Describe your specific use case and how this feature would help
      placeholder: |
        I am working on a project where I need to...
        This feature would help me by...
    validations:
      required: true
      
  - type: textarea
    id: api_design
    attributes:
      label: 🔧 API Design (Optional)
      description: If you have ideas about how this feature should work, share them
      render: python
      placeholder: |
        # Example of how I envision the API:
        from dmdslab import NewFeature
        
        feature = NewFeature()
        result = feature.do_something()
        
  - type: textarea
    id: additional
    attributes:
      label: 📝 Additional Context
      description: Add any other context, mockups, links, or examples
      placeholder: Links to similar features in other libraries, research papers, etc.
      
  - type: checkboxes
    id: contribution
    attributes:
      label: 🤝 Contribution
      description: Are you willing to help implement this feature?
      options:
        - label: I would be willing to implement this feature
          required: false
        - label: I would be willing to test this feature
          required: false
        - label: I would be willing to write documentation for this feature
          required: false
          
  - type: checkboxes
    id: terms
    attributes:
      label: ✅ Checklist
      description: Please confirm the following
      options:
        - label: I have searched for existing feature requests that might be similar
          required: true
        - label: This feature aligns with DmDSLab's goals (data science research tools)
          required: true