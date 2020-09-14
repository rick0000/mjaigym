Gem::Specification.new do |s|
  
    s.name = "mjai"
    s.version = "0.0.7"
    s.authors = ["Hiroshi Ichikawa"]
    s.email = ["gimite+github@gmail.com"]
    s.summary = "Game server for Japanese Mahjong AI."
    s.description = "Game server for Japanese Mahjong AI."
    s.homepage = "https://github.com/gimite/mjai"
    s.license = "New BSD"
    s.rubygems_version = "1.2.0"
    
    s.files = Dir["bin/*"] + Dir["lib/**/*"] + Dir["share/**/*"]
    s.require_paths = ["lib"]
    s.executables = Dir["bin/*"].map(){ |pt| File.basename(pt) }
    s.has_rdoc = true
    s.extra_rdoc_files = []
    s.rdoc_options = []
  
    s.add_dependency("json", ["2.3.1"])
    s.add_dependency("nokogiri", ["1.10.10"])
    s.add_dependency("bundler", ["2.1.4"])
    s.add_dependency("sass", ["3.7.4"])
    
  end