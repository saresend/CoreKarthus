
import Foundation


/*
    A Tool spec that is designed to work with Open AIs tool spec
 */


public struct ToolDescription: Codable {
    let type: String
    let name: String
    let description: String
    let parameters: ToolParameters
}

public struct ToolParameters: Codable {
    let type: String
    let properties: ToolProperties
    let required: [String]
}

public struct ToolProperties: Codable {
    let sign: ToolPropertyDetail
}

public struct ToolPropertyDetail: Codable {
    let type: String
    let description: String

    public init(type: String, description: String) {
        self.type = type
        self.description = description
    }
}


